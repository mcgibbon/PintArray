# -*- coding: utf-8 -*-
"""Main module."""
# -*- coding: utf-8 -*-
import pint
import xarray as xr
import operator
from six import string_types


class PintArrayUnitRegistry(pint.UnitRegistry):

    def __call__(self, input_string, **kwargs):
        return super(UnitRegistry, self).__call__(
            # These replacements are necessary because Pint produces errors
            # if you give it these characters.
            input_string.replace(
                u'%', 'percent').replace(
                u'°', 'degree'
            ),
            **kwargs)

    def Quantity(self, object, units):
        return Quantity(object, units)


unit_registry = PintArrayUnitRegistry()
unit_registry.define('degrees_north = degree_north = degree_N = degrees_N = degreeN = degreesN')
unit_registry.define('degrees_east = degree_east = degree_E = degrees_E = degreeE = degreesE')
unit_registry.define('percent = 0.01*count')


def is_valid_unit(unit_string):
    """Returns True if the unit string is recognized, and False otherwise."""
    # These replacements are necessary because Pint produces errors if you
    # give it these characters.
    unit_string = unit_string.replace(
        '%', 'percent').replace(
        '°', 'degree')
    try:
        unit_registry(unit_string)
    except pint.UndefinedUnitError:
        return False
    else:
        return True


def data_array_to_units(value, to_units, inplace=False):
    if not hasattr(value, 'attrs') or 'units' not in value.attrs:
        raise TypeError(
            'Cannot retrieve units from type {}'.format(type(value)))
    elif unit_registry(value.attrs['units']) == unit_registry(to_units):
        return value
    else:
        if inplace:
            new = value
        else:
            new = value.copy(deep=True)
        from_units = value.attrs['units']
        unit_registry.convert(new.values, from_units, to_units, inplace=True)
        new.attrs['units'] = str(to_units)
        return new


def from_unit_to_another(value, original_units, new_units):
    return (unit_registry(original_units)*value).to(new_units).magnitude


class UnitContainer():
    """
    Contains units information, which interacts with a PintArray the same
    way a pint.UnitContainer interacts with a pint.Quantity.
    """
    pass


def Quantity(object, units):
    return PintArray(object, attrs={'units': str(units)})


def UnitRegistry():
    return unit_registry


def get_dimensionality(pint_array):
    return unit_registry.get_dimensionality(pint_array.attrs['units'])


def has_compatible_delta(pint_array, unit):
    """
    Check if pint_array has a delta_ unit that is compatible with unit.
    """
    deltas = get_delta_units(pint_array)
    if 'delta_' + unit in deltas:
        return True
    else:  # Look for delta units with same dimension as the offset unit
        offset_unit_dim = unit_registry(unit).reference
        for d in deltas:
            if unit_registry(d).reference == offset_unit_dim:
                return True
    return False


class PintArray(xr.DataArray, pint.Quantity):

    _REGISTRY = unit_registry

    # TODO: make add and sub work with non-multiplicative units
    #       e.g. degC, degK, delta_degK

    def __add__(self, other):
        return self._add_sub(other, operator.add)

    __radd__ = __add__

    def __sub__(self, other):
        return self._add_sub(other, operator.sub)

    def __rsub__(self, other):
        return -self._add_sub(other, operator.sub)

    def __iadd__(self, other):
        return self._add_sub(other, operator.iadd)

    def __isub__(self, other):
        return self._add_sub(other, operator.isub)

    def _add_sub(self, other, op):
        if isinstance(other, xr.DataArray) and ('units' in other.attrs):
            if not get_dimensionality(self) == get_dimensionality(other):
                raise pint.DimensionalityError(
                    self._units, unit_registry(other.attrs['units']),
                    get_dimensionality(self),
                    get_dimensionality(other),
                )

            self_non_mul_units = get_non_multiplicative_units(self)
            self_is_multiplicative = len(self_non_mul_units) == 0
            other_non_mul_units = get_non_multiplicative_units(other)
            other_is_multiplicative = len(other_non_mul_units) == 0
            if self_is_multiplicative and other_is_multiplicative:
                if self._units == unit_registry(other.attrs['units']):
                    array = op(self, other)
                    array.attrs['units'] = self.attrs['units']
                elif get_delta_units(self) and not get_delta_units(other):
                    array = op(self.to(other.attrs['units']), other)
                    array.attrs['units'] = other.attrs['units']
                else:
                    array = op(
                        self,
                        data_array_to_units(other, self._units),
                    )
                    array.attrs['units'] = self.attrs['units']
            elif (
                    op == operator.sub and
                    len(self_non_mul_units) == 1 and
                    self._units[self_non_mul_units[0]] == 1 and
                    not has_compatible_delta(other, self_non_mul_units[0])):
                if self._units == unit_registry(other.attrs['units']):
                    array = op(self, other)
                    array.attrs['units'] = self.attrs['units']
                else:
                    array = op(
                        self,
                        data_array_to_units(other, self._units)
                    )
                    array.attrs['units'] = self.attrs['units']
            elif (
                    op == operator.sub and
                    len(other_non_mul_units) == 1 and
                    unit_registry(other.attrs['units'])[other_non_mul_units[0]] == 1 and
                    not has_compatible_delta(self, other_non_mul_units[0])):
                array = op(self, data_array_to_units(other, self._units))
                array.attrs['units'] = self.attrs['units']
            elif (
                    len(self_non_mul_units) == 1 and
                    self._units[self_non_mul_units[0]] == 1 and
                    has_compatible_delta(other, self_non_mul_units[0])):
                to_units = self._units.rename(
                    self_non_mul_units[0], 'delta_' + self_non_mul_units[0])
                array = op(self, data_array_to_units(other, to_units))
                array.attrs['units'] = self.attrs['units']
            elif (
                    len(other_non_mul_units) == 1 and
                    unit_registry(other.attrs['units'])[other_non_mul_units[0]] == 1 and
                    has_compatible_delta(self, other_non_mul_units[0])):
                to_units = other._units.rename(
                    other_non_mul_units[0], 'delta_' + other_non_mul_units[0])
                array = op(self.to(to_units), other)
                array.attrs['units'] = other.attrs['units']
            else:
                raise pint.OffsetUnitCalculusError(
                    self._units, unit_registry(other.attrs['units']))
        elif other == 0:
            array = op(self, other)
            array.attrs['units'] = ''
        elif self.dimensionless:
            array = op(self.to(''), other)
            array.attrs['units'] = ''
        else:
            raise pint.DimensionalityError(self._units, 'dimensionless')
        return array

    def to_units(self, units, inplace=False):
        """
        Convert the units of this DataArray, if necessary. No conversion is
        performed if the units are the same as the units of this DataArray.
        The units of this DataArray are determined from the "units" attribute in
        attrs.

        Args
        ----
        units : str
            The desired units.
        inplace : bool
            Whether to perform the conversion on this DataArray, or create
            a new one.

        Raises
        ------
        ValueError
            If the units are invalid for this object.
        KeyError
            If this object does not have units information in its attrs.

        Returns
        -------
        converted_data : DataArray
            A DataArray containing the data from this object in the
            desired units, if possible. If inplace is True, then this
            is the same object as this DataArray.
        """
        if 'units' not in self.attrs:
            raise KeyError('"units" not present in attrs')
        try:
            return data_array_to_units(self, units, inplace=inplace)
        except pint.DimensionalityError as err:
            raise ValueError(str(err))

    @property
    def magnitude(self):
        return self.values

    @property
    def m(self):
        return self.magnitude

    @property
    def unitless(self):
        """Return true if the quantity does not have units.
        """
        raise NotImplementedError()

    @property
    def dimensionality(self):
        return get_dimensionality(self)

    def to(self, other):
        return self.to_units(other)

    def ito(self, units):
        return self.to_units(units, inplace=True)

    def to_root_units(self):
        _, root_units = unit_registry.get_root_units(self._units)
        return self.to_units(root_units)

    def ito_root_units(self):
        _, root_units = unit_registry.get_root_units(self._units)
        return self.ito(root_units)

    def to_base_units(self):
        _, base_units = unit_registry.get_base_units(self._units)
        return self.to(base_units)

    def ito_base_units(self):
        _, base_units = unit_registry.get_base_units(self._units)
        return self.ito(base_units)

    @property
    def _units(self):
        return unit_registry(self.attrs['units'])

    def compatible_units(self):
        return unit_registry.get_compatible_units(self._units)

    def compare(self, other, op):
        if not isinstance(other, self.__class__):
            if self.dimensionless:
                return op(self.to_units(''), other)
            else:
                raise ValueError(
                    'Cannot compare non-dimensionless PintArray'
                    ' and {0}'.format(type(other)))

        if self._units == other._units:
            return op(self, other)
        elif self.dimensionality == other.dimensionality:
            return op(self.to_root_units(), other.to_root_units())
        else:
            raise pint.DimensionalityError(
                self._units, other._units,
                self.dimensionality, other.dimensionality)

    __lt__ = lambda self, other: self.compare(other, op=operator.lt)
    __le__ = lambda self, other: self.compare(other, op=operator.le)
    __ge__ = lambda self, other: self.compare(other, op=operator.ge)
    __gt__ = lambda self, other: self.compare(other, op=operator.gt)
    __eq__ = lambda self, other: self.compare(other, op=operator.eq)
    __ne__ = lambda self, other: self.compare(other, op=operator.ne)

    def _get_delta_units(self):
        return get_delta_units(self)

    def _get_non_multiplicative_units(self):
        return get_non_multiplicative_units(self)


def get_delta_units(pint_array):
    return [u for u in pint_array._units.keys()
            if u.startswith('delta_')]


def get_non_multiplicative_units(pint_array):
    return [u for u in pint_array._units.keys()
            if not unit_registry(u).is_multiplicative]
