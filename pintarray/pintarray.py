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
            input_string.replace(
                u'%', 'percent').replace(
                u'°', 'degree'
            ),
            **kwargs)


unit_registry = PintArrayUnitRegistry()
unit_registry.define('degrees_north = degree_north = degree_N = degrees_N = degreeN = degreesN')
unit_registry.define('degrees_east = degree_east = degree_E = degrees_E = degreeE = degreesE')
unit_registry.define('percent = 0.01*count = %')


def is_valid_unit(unit_string):
    """Returns True if the unit string is recognized, and False otherwise."""
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
        new.attrs['units'] = to_units
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
    if isinstance(object, PintArray):
        raise NotImplementedError()
    elif isinstance(object, xr.DataArray):
        raise NotImplementedError()
    else:
        if not isinstance(units, string_types):
            raise NotImplementedError()
        return PintArray(object, attrs={'units': units})


def Quantity(data, units):
    return PintArray(data, attrs={'units': str(units)})

def UnitRegistry():
    return unit_registry

class PintArray(xr.DataArray, pint.Quantity):

    # TODO: make add and sub work with non-multiplicative units
    #       e.g. delta_degK

    def __add__(self, other):
        if isinstance(other, xr.DataArray) and ('units' in other.attrs):
            other = data_array_to_units(other, self._units)
        return super(PintArray, self).__add__(other)

    def __sub__(self, other):
        if isinstance(other, xr.DataArray) and ('units' in other.attrs):
            other = data_array_to_units(other, self._units)
        return super(PintArray, self).__sub__(other)

    def __iadd__(self, other):
        if isinstance(other, xr.DataArray) and ('units' in other.attrs):
            other = data_array_to_units(other, self._units)
        return super(PintArray, self).__iadd__(other)

    def __isub__(self, other):
        if isinstance(other, xr.DataArray) and ('units' in other.attrs):
            other = data_array_to_units(other, self._units)
        return super(PintArray, self).__isub__(other)

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
        return unit_registry.get_dimensionality(self._units)

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
        return self.attrs['units']

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
