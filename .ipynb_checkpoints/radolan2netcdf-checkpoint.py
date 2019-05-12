#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# Copyright (c) 2018, miub developers.
# Distributed under the MIT License. See LICENSE.txt for more info.

import warnings
warnings.simplefilter('ignore', DeprecationWarning)
warnings.simplefilter('ignore', UserWarning)

import netCDF4 as nc4
import numpy as np
import datetime as dt
import dateutil.parser as dparser
import sys
import os
import glob
import fnmatch
import getopt
import gzip
import argparse
from tqdm import tqdm


import wradlib as wrl

def read_RADOLAN_composite(fname, missing=-9999, loaddata=True):
    """Read quantitative radar composite format of the German Weather Service

    The quantitative composite format of the DWD (German Weather Service) was
    established in the course of the `RADOLAN project <http://www.dwd.de/RADOLAN>`
    and includes several file types, e.g. RX, RO, RK, RZ, RP, RT, RC, RI, RG, PC,
    PG and many, many more.
    (see format description on the RADOLAN project homepage :cite:`DWD2009`).

    At the moment, the national RADOLAN composite is a 900 x 900 grid with 1 km
    resolution and in polar-stereographic projection. There are other grid resolutions
    for different composites (eg. PC, PG)

    **Beware**: This function already evaluates and applies the so-called PR factor which is
    specified in the header section of the RADOLAN files. The raw values in an RY file
    are in the unit 0.01 mm/5min, while read_RADOLAN_composite returns values
    in mm/5min (i. e. factor 100 higher). The factor is also returned as part of
    attrs dictionary under keyword "precision".

    Parameters
    ----------
    fname : path to the composite file

    missing : value assigned to no-data cells

    Returns
    -------
    output : tuple of two items (data, attrs)
        - data : numpy array of shape (number of rows, number of columns)
        - attrs : dictionary of metadata information from the file header

    """

    NODATA = missing
    mask = 0xFFF  # max value integer

    f = wrl.io.get_radolan_filehandle(fname)

    header = wrl.io.read_radolan_header(f)

    attrs = wrl.io.parse_dwd_composite_header(header)

    if not loaddata:
        f.close()
        return None, attrs

    attrs["nodataflag"] = NODATA

    if not attrs["radarid"] == "10000":
        warnings.warn("WARNING: You are using function e" +
                      "wradlib.io.read_RADOLAN_composit for a non " +
                      "composite file.\n " +
                      "This might work...but please check the validity " +
                      "of the results")

    # read the actual data
    indat = wrl.io.read_radolan_binary_array(f, attrs['datasize'])

    # data handling taking different product types into account
    # RX, EX, WX 'qualitative', temporal resolution 5min, RVP6-units [dBZ]
    if attrs["producttype"] in ["RX", "EX", "WX"]:
        #convert to 8bit unsigned integer
        arr = np.frombuffer(indat, np.uint8).astype(np.uint8)
        # clutter & nodata
        cluttermask = np.where(arr == 249)[0]
        nodatamask = np.where(arr == 250)[0]
        #attrs['cluttermask'] = np.where(arr == 249)[0]

        #arr = np.where(arr >= 249, np.int32(255), arr)

    elif attrs['producttype'] in ["PG", "PC"]:
        arr = wrl.io.decode_radolan_runlength_array(indat, attrs)
    else:
        # convert to 16-bit integers
        arr = np.frombuffer(indat, np.uint16).astype(np.uint16)
        # evaluate bits 13, 14, 15 and 16
        secondary = np.where(arr & 0x1000)[0]
        attrs['secondary'] = np.where(arr & 0x1000)[0]
        #attrs['nodata'] = np.where(arr & 0x2000)[0]
        nodatamask = np.where(arr & 0x2000)[0]
        negative = np.where(arr & 0x4000)[0]
        cluttermask = np.where(arr & 0x8000)[0]
        #attrs['cluttermask'] = np.where(arr & 0x8000)[0]

        # mask out the last 4 bits
        arr = arr & mask

        # consider negative flag if product is RD (differences from adjustment)
        if attrs["producttype"] == "RD":
            # NOT TESTED, YET
            arr[negative] = -arr[negative]
        # apply precision factor
        # this promotes arr to float if precision is float
        #arr = arr * attrs["precision"]
        # set nodata value#
        #arr[attrs['secondary']] = np.int32(4096)
        #arr[nodata] = np.int32(4096)#NODATA

    if nodatamask is not None:
        attrs['nodatamask'] = nodatamask
    if cluttermask is not None:
        attrs['cluttermask'] = cluttermask
    #arr[np.where(arr == 2500)[0]] = np.int32(4096)
    #arr[np.where(arr == 2490)[0]] = np.int32(4096)
    #arr[nodata] = np.int32(0)
    #arr[clutter] = np.int32(65535)
    # anyway, bring it into right shape
    arr = arr.reshape((attrs["nrow"], attrs["ncol"]))
    #arr = arr.reshape((attrs["nrow"], attrs["ncol"]))

    return arr, attrs


def create_ncdf(fname, attrs, units='original'):

    nx = attrs['ncol']
    ny = attrs['nrow']

    # create NETCDF4 file and close again
    id = nc4.Dataset(fname, 'w', format='NETCDF4')
    id.close()

    # open NETCDF4 file for appending data
    id = nc4.Dataset(fname, 'a', format='NETCDF4')

    # create dimensions
    yid = id.createDimension('y', ny)
    xid = id.createDimension('x', nx)
    tbid = id.createDimension('nv', 2)
    tid = id.createDimension('time', None)

    # create and set the grid x variable that serves as x coordinate
    xiid = id.createVariable('x', 'f4', ('x'))
    xiid.axis = 'X'
    xiid.units = 'km'
    xiid.long_name = 'x coordinate of projection'
    xiid.standard_name = 'projection_x_coordinate'

    # create and set the grid y variable that serves as y coordinate
    yiid = id.createVariable('y', 'f4', ('y'))
    yiid.axis = 'Y'
    yiid.units = 'km'
    yiid.long_name = 'y coordinate of projection'
    yiid.standard_name = 'projection_y_coordinate'

    # create time variable
    tiid = id.createVariable('time', 'f8', ('time',))
    tiid.axis = 'T'
    tiid.units = 'seconds since 1970-01-01 00:00:00'
    tiid.standard_name = 'time'
    tiid.bounds = 'time_bnds'

    # create time bounds variable
    tbiid = id.createVariable('time_bnds', 'f8', ('time', 'nv',))

    # create grid variable that serves as lon coordinate
    lonid = id.createVariable('lon', 'f4', ('y', 'x',), zlib=True, complevel=4)
    lonid.units = 'degrees_east'
    lonid.standard_name = 'longitude'
    lonid.long_name = 'longitude coordinate'

    # create grid variable that serves as lat coordinate
    latid = id.createVariable('lat', 'f4', ('y', 'x',), zlib=True, complevel=4)
    latid.units = 'degrees_north'
    latid.standard_name = 'latitude'
    latid.long_name = 'latitude coordinate'

    # create projection variable that defines the projection according to CF-Metadata standards
    coordid = id.createVariable('polar_stereographic', 'i4', zlib=True, complevel=4)
    coordid.grid_mapping_name = 'polar_stereographic'
    coordid.straight_vertical_longitude_from_pole = np.float32(10.)
    coordid.latitude_of_projection_origin = np.float32(90.)
    coordid.standard_parallel = np.float32(60.)
    coordid.false_easting = np.float32(0.)
    coordid.false_northing = np.float32(0.)
    coordid.earth_model_of_projection = 'spherical'
    coordid.earth_radius_of_projection = np.float32(6370.04)
    coordid.units = 'km'
    coordid.ancillary_data = 'grid_latitude grid_longitude'
    coordid.long_name = 'polar_stereographic'

    # create the data variable
    # set zlib compression to reasonable level
    version = attrs['radolanversion']
    precision = attrs['precision']
    prodtype = attrs['producttype']
    int = attrs['intervalseconds']
    nodata = attrs['nodataflag']
    missing_value = None

    if prodtype in ['RX', 'EX']:
        if units == 'original':
            scale_factor = None
            add_offset = None
            unit = 'RVP6'
        else:
            scale_factor = np.float32(0.5)
            add_offset = np.float32(-32.5)
            unit = 'dBZ'

        valid_min = np.int32(0)
        valid_max = np.int32(255)
        missing_value = np.int32(255)
        fillvalue = np.int32(255)
        vtype = 'u1'
        standard_name = 'equivalent_reflectivity_factor'
        long_name = 'equivalent_reflectivity_factor'

    elif prodtype in ['RY', 'RZ', 'EY', 'EZ']:
        if units == 'original':
            scale_factor = None
            add_offset = None
            unit = '0.01mm 5min-1'
        elif units == 'normal':
            scale_factor = np.float32(precision * 3600/int)
            add_offset = np.float(0)
            unit = 'mm h-1'
        else:
            scale_factor = np.float32(precision/(int*1000))
            add_offset = np.float(0)
            unit = 'm s-1'

        valid_min = np.int32(0)
        valid_max = np.int32(4095)
        missing_value = np.int32(4096)
        fillvalue = np.int32(65535)
        vtype = 'u2'
        standard_name = 'rainfall_amount'
        long_name = 'rainfall_amount'

    elif prodtype in ['RH', 'RB', 'RW', 'RL', 'RU', 'EH', 'EB', 'EW']:
        if units == 'original':
            scale_factor = None
            add_offset = None
            unit = '0.1mm h-1'
        elif units == 'normal':
            scale_factor = np.float32(precision)
            add_offset = np.float(0.)
            unit = 'mm h-1'
        else:
            scale_factor = np.float32(precision/(int*1000))
            add_offset = np.float(0)
            unit = 'm s-1'

        valid_min = np.int32(0)
        valid_max = np.int32(4095)
        missing_value = np.int32(4096)
        fillvalue = np.int32(65535)
        vtype = 'u2'
        standard_name = 'rainfall_amount'
        long_name = 'rainfall_amount'

    elif prodtype in ['SQ', 'SH', 'SF']:
        scale_factor = np.float32(precision)
        add_offset = np.float(0.)
        valid_min = np.int32(0)
        valid_max = np.int32(4095)
        missing_value = np.int32(4096)
        fillvalue = np.int32(65535)
        vtype = 'u2'
        standard_name = 'rainfall_amount'
        long_name = 'rainfall_amount'
        if int == (360 * 60):
            unit = 'mm 6h-1'
        elif int == (720 * 60):
            unit = 'mm 12h-1'
        elif int == (1440 * 60):
            unit = 'mm d-1'

    id_product = id.createVariable(prodtype.lower(), vtype,
                                   ('time', 'y', 'x',), fill_value=fillvalue,
                                   zlib=True, complevel=4)
    # accept data as unsigned byte without scaling, crucial for writing already packed data
    id_product.set_auto_maskandscale(False)
    id_product.units = unit
    id_product.standard_name = standard_name
    id_product.long_name = long_name
    id_product.grid_mapping = 'polar_stereographic'
    id_product.coordinates = 'lat lon'
    if scale_factor:
        id_product.scale_factor = scale_factor
    if add_offset:
        id_product.add_offset = add_offset
    if valid_min:
        id_product.valid_min = valid_min
    if valid_max:
        id_product.valid_max = valid_max
    if missing_value:
        id_product.missing_value = missing_value
    id_product.version = 'RADOLAN {0}'.format(version)
    id_product.comment = 'NO COMMENT'

    id_str1 = id.createVariable('radars', 'S128', ('time',),
                                zlib=True, complevel=4)

    # create GLOBAL attributes
    id.Title = 'RADOLAN {0} Composite'.format(prodtype)
    id.Institution = 'Data owned by Deutscher Wetterdienst'
    id.Source = 'DWD C-Band Weather Radar Network, Original RADOLAN Data by ' \
                'Deutscher Wetterdienst'
    id.History = 'Data transferred from RADOLAN composite format to netcdf ' \
                 'using "miubrt" version {0} ' \
                 'by Uni Bonn'
    id.Conventions = 'CF-1.6 where applicable'
    utcnow = dt.datetime.utcnow()
    id.Processing_date = utcnow.strftime("%Y-%m-%dT%H:%M:%S")
    id.Author = 'MIUB Radar, radar@uni-bonn.de'
    id.Comments = 'blank'
    id.License = 'DWD Licenses'

    return id


def radolan2netcdf(inpath, outpath, dyear, dmonth, dday, units):

    # gather data, fill netcdf variables
    disable_year = len(dyear) == 1
    disable_month = len(dmonth) == 1
    disable_day = len(dday) == 1
    for year in tqdm(dyear, 'y', ascii=True, disable=disable_year):
        for month in tqdm(dmonth, 'm', ascii=True, disable=disable_month):
            for day in tqdm(dday, 'd', ascii=True, disable=disable_day):
                try:
                    dt.date(year, month, day)
                except:
                    continue

                time_index = 0

                radolan_filename = create_radolan_filename(year, month, day)
                print(radolan_filename)
                files_found = find_files(inpath, pattern=radolan_filename)

                for f in tqdm(list(files_found), 'f', ascii=True):
                    product, attrs = read_RADOLAN_composite(f)

                    if product is not None:
                        if time_index == 0:
                            ny, nx = attrs['ncol'],attrs['nrow']
                            radolan_grid_xy = wrl.georef.get_radolan_grid(nx, ny)
                            xarr = radolan_grid_xy[0,:,0]
                            yarr = radolan_grid_xy[:,0,1]
                            radolan_grid_ll = wrl.georef.get_radolan_grid(nx, ny, wgs84=True)
                            lons = radolan_grid_ll[...,0]
                            lats = radolan_grid_ll[...,1]
                            epoch = dt.datetime.utcfromtimestamp(0)

                            id = create_ncdf(outpath  + '{0}-{1}-{2:02}-{3:02}.nc'.format(attrs['producttype'],
                                                                                          year, month, day),
                                             attrs, units)
                            id.variables['x'][:] = xarr
                            id.variables['x'].valid_min = xarr[0]
                            id.variables['x'].valid_max = xarr[-1]
                            id.variables['y'][:] = yarr
                            id.variables['y'].valid_min = yarr[0]
                            id.variables['y'].valid_max = yarr[-1]
                            id.variables['lat'][:] = lats
                            id.variables['lon'][:] = lons

                        # remove clutter, nodata and secondary data from raw files
                        # wrap with if/else if necessary
                        if attrs['cluttermask'] is not None:
                            product.flat[attrs['cluttermask']] = id.variables[attrs['producttype'].lower()].missing_value
                        if attrs['nodatamask'] is not None:
                            product.flat[attrs['nodatamask']] = id.variables[attrs['producttype'].lower()].missing_value
                        #if attrs['secondary'] is not None:
                        #    zhbytes.flat[attrs['secondary']] = id.variables[attrs['producttype'].lower()].missing_value

                        id.variables[attrs['producttype'].lower()][time_index,:,:] = product
                        delta = attrs['datetime'] - epoch
                        id.variables['time'][time_index] = delta.total_seconds()
                        id.variables['time_bnds'][time_index,:] = delta.total_seconds()
                        id.variables['radars'][time_index] = ','.join(attrs['radarlocations'])
                        time_index = time_index + 1

                if time_index == 0:
                    print("No valid datafiles found for Y-m-d: {0}-{1:02}-{2:02} below folder {3}".format(year, month, day, inpath))
                    continue

                id.close()


def find_files(folder, pattern, func=None, **funcargs):
    print(folder)
    for root, dirs, files in os.walk(folder):
        print(files)
        for basename in sorted(files):
            print(basename)
            if fnmatch.fnmatch(basename, pattern):
                if func:
                    if func(basename, **funcargs):
                        filename = os.path.join(root, basename)
                        yield filename
                else:
                    filename = os.path.join(root, basename)
                    yield filename


def create_radolan_filename(year, month, day):
    return 'raa01-{0}_10000-{1}{2:02}{3:02}'.format('*', year-2000,month,day) + '*--bin*'


def expand_numbers(number_string):
    ret = []
    for sc in number_string.split(','):
        if sc.isdigit():
            ret.extend(np.int16(sc.split()))
        else:
            ss = sc.split('-')
            ret.extend(list(np.arange(np.int16(ss[0]), np.int16(ss[1])+1, 1)))
    keys = {}
    for e in ret:
       keys[e] = 1
    return sorted(keys.keys())


def import_dates(date_string):
    return dparser.parse(date_string)


def main():

    parser = argparse.ArgumentParser(description='Convert Radolan Composite to NetCDF', add_help=True)

    d1group = parser.add_argument_group(description='specific date parameters')
    d1group.add_argument('-y', '--year', nargs='*', type=expand_numbers, required=True)
    d1group.add_argument('-m', '--month', nargs='*', type=expand_numbers)
    d1group.add_argument('-d', '--day', nargs='*', type=expand_numbers)

    iogroup = parser.add_argument_group(description='input and output folders, mandatory')
    iogroup.add_argument('-i', '--input-folder', required=True)
    iogroup.add_argument('-o', '--output-folder', required=True)

    parser.add_argument('-u', '--units', default='normal')

    pargs = parser.parse_args()

    if pargs.day and pargs.month is None:
        parser.error("-d [--day] requires -m [--month] and -y [--year].")

    if pargs.month and pargs.year is None:
        parser.error("-m [--month] requires -y [--year].")

    if pargs.year:
        year = pargs.year[0]

    if pargs.month:
        month = pargs.month[0]
    else:
        month = np.arange(1,13,1)

    if pargs.day:
        day = pargs.day[0]
    else:
        day = np.arange(1,32,1)

    inpath = pargs.input_folder
    outpath = pargs.output_folder

    units = pargs.units

    radolan2netcdf(inpath, outpath, year, month, day, units)


if __name__ == '__main__':
    main()

