#!/usr/bin/env python
# -*- coding: utf-8 -*-

import gc
import os
import os.path as op
import warnings
from collections.abc import Mapping

import xarray as xr
import numpy as np



def store_xarray_max_compression(dataset, p_store):
    """Store an xarray dataset using maximum NetCDF compression."""
    compression = dict(zlib=True, complevel=9)
    encoding = {variable: compression for variable in dataset.data_vars}
    dataset.to_netcdf(p_store, encoding=encoding)


def _normalise_sources(spectra):
    """Return ``[(point_name, source), ...]`` from a mapping or sequence.

    A source can be a NetCDF path or an already opened ``xarray.Dataset``.
    Mapping keys are deliberately independent from file names so callers can use
    any point naming convention.
    """
    if isinstance(spectra, Mapping):
        sources = list(spectra.items())
    else:
        sources = [(op.splitext(op.basename(op.fspath(source)))[0], source)
                   for source in spectra]
    if not sources:
        raise ValueError("At least one wave spectrum must be provided")
    return sources


def _open_spectrum(source):
    if isinstance(source, xr.Dataset):
        return source, False
    return xr.open_dataset(source), True


def fix_dir(base_dirs):
    '''
    fix csiro direction for wavespectra (from -> to)'
    '''

    new_dirs = base_dirs + 180
    new_dirs[np.where(new_dirs >= 360)] = new_dirs[np.where(new_dirs >= 360)] - 360

    return new_dirs


def validate_directional_sectors(sectors, directions=None, resolution=0.5):
    """Warn about gaps or duplicated angles in directional sectors.

    Sectors use the nautical convention and are interpreted as ``(start, end]``
    so a shared boundary between two adjacent sectors is counted only once.
    Validation concerns the base sectors only; the intentional overlap added by
    ``deg_sup`` during SuperPoint construction is not included.

    Parameters
    ----------
    sectors : mapping
        Mapping of point names to ``(start, end)`` angles in degrees.
    directions : array-like, optional
        Angles to validate. If omitted, the complete circle is sampled using
        ``resolution``.
    resolution : float, default 0.5
        Angular sampling interval in degrees when ``directions`` is omitted.

    Returns
    -------
    numpy.ndarray
        Number of sectors representing each validated direction.
    """
    if not isinstance(sectors, Mapping) or not sectors:
        raise ValueError("sectors must be a non-empty mapping")
    if resolution <= 0 or resolution >= 360:
        raise ValueError("resolution must be between 0 and 360 degrees")

    angles = (np.arange(0, 360, resolution) if directions is None
              else np.mod(np.asarray(directions, dtype=float), 360))
    coverage = np.zeros(angles.size, dtype=int)
    contributors = [[] for _ in range(angles.size)]

    for point_name, sector in sectors.items():
        if len(sector) != 2:
            raise ValueError(
                "Sector for point '{0}' must contain start and end angles"
                .format(point_name)
            )
        start, end = map(float, sector)
        width = (end - start) % 360
        if width == 0:
            warnings.warn(
                "Sector for point '{0}' has zero angular width"
                .format(point_name), UserWarning, stacklevel=2,
            )
            continue
        offset = np.mod(angles - (start % 360), 360)
        represented = (offset > 0) & (offset <= width)
        coverage[represented] += 1
        for index in np.flatnonzero(represented):
            contributors[index].append(str(point_name))

    uncovered = np.flatnonzero(coverage == 0)
    if uncovered.size:
        examples = ', '.join('{0:g}°'.format(angles[i]) for i in uncovered[:8])
        warnings.warn(
            "Directional sectors leave {0} checked angle(s) unrepresented"
            " (examples: {1}).".format(uncovered.size, examples),
            UserWarning, stacklevel=2,
        )

    duplicated = np.flatnonzero(coverage > 1)
    if duplicated.size:
        examples = '; '.join(
            '{0:g}°: {1}'.format(angles[i], ', '.join(contributors[i]))
            for i in duplicated[:8]
        )
        warnings.warn(
            "Directional sectors represent {0} checked angle(s) more than"
            " once (examples: {1}).".format(duplicated.size, examples),
            UserWarning, stacklevel=2,
        )

    return coverage

def stations_superposition(spectra, sectors, deg_sup, wind_point,
                           fix_dir_bool = True, efth_to_rad = True,
                           freq_n='frequency', dir_n='direction', efth_n='Efth',
                           time_n='time', wspeed_n='u10m', wdir_n='udir',
                           depth_n='depth'):
    '''
    Join station spectral data for each sector

    spectra      - mapping ``{point_name: netcdf_path_or_dataset}``; file names
                   and point names are arbitrary. A sequence of paths is also
                   accepted and uses each file stem as its point name.
    sectors      - mapping ``{point_name: (start, end)}``, or a sequence in the
                   same order as ``spectra``
    deg_sup      - degrees of superposition
    wind_point   - point name from which wind and depth data are read

    fix_dir_bool - fix csiro directions
    efth_to_rad  - transform efth to radians

    freq_n       - name of frequency dimension
    dir_n        - name of direction dimension
    efth_n       - name of efth variable
    wspeed_n     - name of wind speed variable
    wdir_n       - name of wind direction variable
    '''

    sources = _normalise_sources(spectra)
    point_names = [name for name, _ in sources]
    if len(sectors) != len(sources):
        raise ValueError("One directional sector is required per spectrum")
    if isinstance(sectors, Mapping) and set(sectors) != set(point_names):
        raise ValueError("Sector names must match spectrum point names")
    if wind_point not in point_names:
        raise ValueError("wind_point must be one of: {0}".format(point_names))

    reference, close_reference = _open_spectrum(sources[0][1])
    try:
        time = reference[time_n].values
        frequency = reference[freq_n].values
        direction = reference[dir_n].values
    finally:
        if close_reference:
            reference.close()

    output_direction = fix_dir(direction) if fix_dir_bool else direction
    sectors_by_point = (sectors if isinstance(sectors, Mapping)
                        else dict(zip(point_names, sectors)))
    validate_directional_sectors(sectors_by_point, directions=output_direction)

    efth_all = np.zeros((len(time), len(frequency), len(direction), len(sources)))
    cont = np.zeros(len(direction), dtype=int)
    wsp = wdir = depth = None

    # read stations
    for s_ix, (point_name, source) in enumerate(sources):
        print('Point: {0}'.format(point_name))
        st, close_station = _open_spectrum(source)
        station_direction = fix_dir(st[dir_n].values) if fix_dir_bool else st[dir_n].values

        if (not np.array_equal(st[time_n].values, time) or
                not np.array_equal(st[freq_n].values, frequency) or
                not np.array_equal(station_direction, output_direction)):
            if close_station:
                st.close()
            raise ValueError("Spectrum coordinates do not match at point '{0}'".format(point_name))

        sector = sectors[point_name] if isinstance(sectors, Mapping) else sectors[s_ix]
        # find spectrum indexes inside sector (and superposition degrees)
        if (sector[1] - sector[0]) < 0:
            d = np.where((station_direction > sector[0] - deg_sup) |
                         (station_direction <= sector[1] + deg_sup))[0]
        else:
            d = np.where((station_direction > sector[0] - deg_sup) &
                         (station_direction <= sector[1] + deg_sup))[0]

        cont[d] += 1
        efth_all[:, :, d, s_ix] = st[efth_n].transpose(time_n, freq_n, dir_n).values[:, :, d]

        # get wind data from choosen wind station
        if point_name == wind_point:
            wsp = st[wspeed_n].values
            wdir = st[wdir_n].values
            depth_values = st[depth_n].values
            depth = (np.full(len(time), depth_values.item()) if depth_values.ndim == 0
                     else np.asarray(depth_values))
        if close_station:
            st.close()

    # promediate superimposed station data (using data counter)
    if np.any(cont == 0):
        raise ValueError("Directional sectors leave directions without a contributing spectrum")
    efth_all = np.sum(efth_all, axis=3) / cont
    if efth_to_rad:
        efth_all = efth_all * (np.pi / 180)

    # mount superpoint dataset
    super_point = xr.Dataset(
        {
            'efth': (['time','freq','dir'], efth_all),
            'Wspeed': (['time'], wsp),
            'Wdir': (['time'], wdir),
            'Depth': (['time'], depth),
        },
        coords = {
            'time': time,
            'dir': output_direction,
            'freq': frequency,
        }
    )
    super_point.attrs['source_points'] = ', '.join(map(str, point_names))

    # round time to hour
    super_point['time'] = super_point['time'].dt.round('h').values

    return super_point


def spectra_superposition(*args, **kwargs):
    """Descriptive alias for :func:`stations_superposition`."""
    return stations_superposition(*args, **kwargs)

def bulkparams_partitions(p_store, sp, chunks=3, wcut=0.333, msw=5, agef=1.7):
    '''
    Calculates superpoint spectra statistics and bulk parameters using wavespectra library.

    p_store - path for storage chunk datasets
    sp      - superpoint Dataset
    chunks  - split process in N chunks (split by time dimension to prevent memory issues)

    wcut    - wavespectra: wind cut
    msw     - wavespectra: max number of swells
    agef    - wavespectra: age factor
    '''

    # ensure storage folder exists
    if not op.isdir(p_store):
        os.makedirs(p_store)

    # this function needs wavespectra==3.5 
    import wavespectra

    # get split position
    pos = int(len(sp.time) / chunks)

    # solve each chunk
    for p in range(chunks):

        # select current chunk  superpoint data
        if p == 0:
            sp1 = sp.isel(time = np.arange(0, pos))

        elif p == (chunks - 1):
            sp1 = sp.isel(time = np.arange(p*pos, len(sp.time)))

        else:
            sp1 = sp.isel(time = np.arange(p*pos, (p+1)*pos))

        # Wavespectra <=3 exposed ``partition`` as a callable, whereas >=4
        # exposes a partition accessor with named methods (PTM1 is equivalent
        # to the former wind-sea plus swell partitioning used here).
        partitioner = sp1.spec.partition
        if callable(partitioner):
            ds_part1 = partitioner(
                sp1.Wspeed, sp1.Wdir, sp1.Depth,
                wscut=wcut, max_swells=msw, agefac=agef,
            )
        else:
            ds_part1 = sp1.efth.spec.partition.ptm1(
                wspd=sp1.Wspeed, wdir=sp1.Wdir, dpt=sp1.Depth,
                wscut=wcut, swells=msw, agefac=agef,
            )

        # clean memory
        del sp1
        gc.collect()

        # ensure time dimension is ok
        _, i = np.unique(ds_part1.time, return_index=True)
        ds_part1 = ds_part1.isel(time=i)

        # store solved chunk spectra
        nf = 'partitions_spectra_chunk_{0}_wcut_{1}.nc'.format(p+1, wcut)
        if isinstance(ds_part1, xr.DataArray):
            ds_part1 = ds_part1.to_dataset(name=ds_part1.name or 'efth')
        store_xarray_max_compression(ds_part1, op.join(p_store, nf))

        # calculate spectral stats
        stats_part1 = ds_part1.spec.stats(['hs','tp','tm02','dpm','dspr'])

        # store spectral stats
        nf = 'partitions_stats_chunk_{0}_wcut_{1}.nc'.format(p+1, wcut)
        stats_part1.to_netcdf(op.join(p_store, nf))

        print('chunk {0}/{1} done.'.format(p+1, chunks))

        # clean memory
        del ds_part1, stats_part1
        gc.collect()


    # load processed chunks spectra stats and merge it 
    nfs = ['partitions_stats_chunk_{0}_wcut_{1}.nc'.format(p+1, wcut) for p in range(chunks)]
    stats_part = xr.open_mfdataset([op.join(p_store, f) for f in nfs])

    # calculate superpoint bulk parameters
    bulk_params = sp.spec.stats(['hs','tp','tm02','dpm','dm','dspr'])

    return bulk_params, stats_part


# Plotting utilities ---------------------------------------------------------

def colormap(elev, topat, topag):
    """Return the bathymetry/topography colormap used by SuperPoint maps."""
    import cmocean
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    bottom = plt.get_cmap('YlGnBu_r', -topag)
    top = plt.get_cmap(cmocean.cm.turbid, topat)
    colors = np.vstack((
        bottom(np.linspace(0, 0.8, -topag)),
        top(np.linspace(0.1, 1, topat)),
    ))
    return ListedColormap(colors)


def load_coast(coast_path=None):
    """Load a coastline shapefile as a Cartopy ``ShapelyFeature``."""
    import cartopy.crs as ccrs
    from cartopy.feature import ShapelyFeature
    from cartopy.io.shapereader import Reader

    if coast_path is None:
        resources = op.join(op.dirname(__file__), 'resources')
        coast_path = op.join(
            resources, 'gshhg-shp-2.3.7', 'GSHHS_shp', 'f', 'GSHHS_f_L1.shp'
        )
    return ShapelyFeature(
        Reader(coast_path).geometries(), ccrs.PlateCarree(), edgecolor='black'
    )


def plot_points(point_coordinates, site_coordinates, point_sectors=None,
                extra_area=1, figsize=(10, 10), sector_radius=None):
    """Plot named spectrum points, their directional sectors and target site.

    ``point_sectors`` follows the nautical convention: directions are measured
    clockwise from north. For example, ``(240, 360)`` represents the sector
    from southwest through north.

    ``extra_area`` is the margin, in degrees, added around the minimum and
    maximum station coordinates.
    """
    import cartopy.feature as cfeature
    import cartopy.crs as ccrs
    import matplotlib.pyplot as plt
    from matplotlib.patches import Wedge

    if not isinstance(point_coordinates, Mapping) or not point_coordinates:
        raise ValueError("point_coordinates must be a non-empty mapping")
    if point_sectors is not None and set(point_sectors) != set(point_coordinates):
        raise ValueError("point_sectors and point_coordinates must use the same names")
    normalized_coordinates = {
        name: (longitude - 360 if longitude > 180 else longitude, latitude)
        for name, (longitude, latitude) in point_coordinates.items()
    }
    lon_site, lat_site = site_coordinates
    lon_site = lon_site - 360 if lon_site > 180 else lon_site
    longitudes = [coordinates[0] for coordinates in normalized_coordinates.values()]
    latitudes = [coordinates[1] for coordinates in normalized_coordinates.values()]
    sector_radius = min(extra_area * 0.5, 0.42) if sector_radius is None else sector_radius
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(
        1, 1, 1, projection=ccrs.PlateCarree(central_longitude=180)
    )
    ax.set_extent(
        (min(longitudes)-extra_area, max(longitudes)+extra_area,
         min(latitudes)-extra_area, max(latitudes)+extra_area),
        crs=ccrs.PlateCarree(),
    )
    ax.add_feature(
        cfeature.OCEAN.with_scale('10m'),
        facecolor='#eef5f7', zorder=0,
    )
    ax.add_feature(
        cfeature.LAND.with_scale('10m'),
        facecolor='#d8c77c', edgecolor='none', zorder=1,
    )
    ax.coastlines(
        resolution='10m', color='#8f7b3e', linewidth=0.8,
        antialiased=True, zorder=3,
    )
    gridlines = ax.gridlines(
        draw_labels=True, linewidth=0.5, color='0.6', alpha=0.35,
        linestyle='-', zorder=2,
    )
    gridlines.top_labels = False
    gridlines.right_labels = False
    gridlines.xlabel_style = {'size': 10}
    gridlines.ylabel_style = {'size': 10}
    # Keep the geographic grid but remove the rectangular map frame.
    for spine in ax.spines.values():
        spine.set_visible(False)
    for point_name, (longitude, latitude) in normalized_coordinates.items():
        if point_sectors is not None:
            start, end = point_sectors[point_name]
            # Matplotlib uses degrees counter-clockwise from east; convert the
            # nautical clockwise-from-north convention used by wave spectra.
            sector = Wedge(
                (longitude, latitude), sector_radius,
                theta1=90-end, theta2=90-start,
                facecolor=(0.0, 0.85, 1.0, 0.16), edgecolor='#00a9cc',
                linewidth=1.8, zorder=7,
            )
            sector.set_transform(ccrs.PlateCarree())
            ax.add_patch(sector)
        ax.plot(
            longitude, latitude, '.', markersize=16, color='darkmagenta',
            zorder=10, transform=ccrs.PlateCarree(), label='Spectrum points',
        )
        ax.text(
            longitude + 0.02, latitude + 0.02, str(point_name), fontsize=16,
            transform=ccrs.PlateCarree(), color='#333333',
        )
    ax.plot(
        lon_site, lat_site, marker='*', markersize=16, zorder=10,
        markerfacecolor='gold', markeredgecolor='black',
        transform=ccrs.PlateCarree(), label='SuperPoint',
    )
    ax.set_title('SuperPoint input stations and directional sectors', pad=12)
    return fig


def Plot_stations(point_coordinates, site_coordinates, point_sectors=None,
                  extra_area=1, figsize=(10, 10), sector_radius=None):
    """Legacy name for :func:`plot_points`."""
    return plot_points(
        point_coordinates, site_coordinates, point_sectors,
        extra_area, figsize, sector_radius,
    )


def Plot_obstructions_CAWCR(obstr_cawcr, area=None, min_z=-10000,
                            max_z=1000, figsize=(25, 12)):
    """Plot CAWCR bathymetry and obstruction percentages."""
    import cartopy.crs as ccrs
    import matplotlib.pyplot as plt

    area = area or []
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(
        1, 1, 1, projection=ccrs.PlateCarree(central_longitude=180)
    )
    ax.add_feature(
        load_coast(), facecolor='None', edgecolor='black', alpha=0.8,
        linewidth=1.5, zorder=5,
    )
    if area:
        lon_ix = np.where(
            (obstr_cawcr.lon.values > area[0]) &
            (obstr_cawcr.lon.values < area[1])
        )[0]
        lat_ix = np.where(
            (obstr_cawcr.lat.values > area[2]) &
            (obstr_cawcr.lat.values < area[3])
        )[0]
        obstr_cawcr = obstr_cawcr.isel(lon=lon_ix, lat=lat_ix)
        ax.set_extent(area, crs=ccrs.PlateCarree())
    ax.pcolormesh(
        obstr_cawcr.lon, obstr_cawcr.lat, obstr_cawcr.depth / 1000,
        cmap=colormap(obstr_cawcr.depth.values / 1000, max_z, min_z),
        vmax=max_z, vmin=min_z, transform=ccrs.PlateCarree(),
        zorder=1, alpha=0.3,
    )
    obstruction_y = obstr_cawcr.obstructions_y.where(
        obstr_cawcr.obstructions_y != 0
    )
    obstruction_x = obstr_cawcr.obstructions_x.where(
        obstr_cawcr.obstructions_x != 0
    )
    plot_y = ax.pcolormesh(
        obstr_cawcr.lon, obstr_cawcr.lat, obstruction_y, cmap='Purples',
        vmin=0, vmax=100, transform=ccrs.PlateCarree(), zorder=2, alpha=0.6,
    )
    plot_x = ax.pcolormesh(
        obstr_cawcr.lon, obstr_cawcr.lat, obstruction_x, cmap='Reds',
        vmin=0, vmax=100, transform=ccrs.PlateCarree(), zorder=2, alpha=0.6,
    )
    if not area:
        ax.contour(
            obstr_cawcr.lon, obstr_cawcr.lat, obstr_cawcr.mask, 3,
            cmap='viridis', zorder=3, transform=ccrs.PlateCarree(),
        )
    plt.colorbar(plot_y).set_label('Obstruction Y (%)', fontsize=16)
    plt.colorbar(plot_x).set_label('Obstruction X (%)', fontsize=16)
    return fig


def _spectral_colormap():
    """Return the common grey-blue-green-yellow-red spectrum colormap."""
    from matplotlib.colors import LinearSegmentedColormap

    return LinearSegmentedColormap.from_list(
        'superpoint_spectrum',
        [
            (0.000, '#f2f2f2'),
            (0.125, '#cce8ee'),
            (0.250, '#24a6c8'),
            (0.375, '#287bea'),
            (0.500, '#3ee7bd'),
            (0.625, '#55ed83'),
            (0.750, '#f2d45c'),
            (0.875, '#f47a25'),
            (1.000, '#ed0000'),
        ],
    )


def _format_spectral_colorbar(colorbar):
    """Apply the shared ticks and label to a spectral colorbar."""
    colorbar.set_ticks(np.arange(0, 0.201, 0.025))
    colorbar.set_label(r'$\sqrt{S}$')


def axplot_spectrum(ax, x, y, z, vmax=0.2, ylim=0.49, cmap=None,
                    plot_center=False):
    """Plot a directional spectrum in an existing polar axis."""
    x_edges = np.append(x, x[0])
    y_edges = np.append(0 if plot_center else y[-1], y)
    if not plot_center:
        y_edges = np.append(y, y[-1])
    plot = ax.pcolormesh(x_edges, y_edges, np.sqrt(z), vmin=0, vmax=vmax)
    plot.set_cmap(_spectral_colormap() if cmap is None else cmap)
    ax.set_theta_zero_location('N', offset=0)
    ax.set_theta_direction(-1)
    ax.set_ylim(0, ylim)
    return plot


def plot_input_spectra(spectra, point_order=None, time_ix=0, gs_rows=1,
                       gs_cols=1, figsize=(10, 10), freq_n='frequency',
                       dir_n='direction', efth_n='Efth', time_n='time',
                       point_coordinates=None, site_coordinates=None,
                       map_padding=0.5, spectrum_size=0.09, vmax=0.2,
                       cmap=None, point_sectors=None, super_point=None,
                       superpoint_size=0.12):
    """Plot each input spectrum in a grid or at its position on a map.

    Passing ``point_coordinates`` activates map mode and places a polar inset
    at every point. Coordinates must be supplied as
    ``{point_name: (longitude, latitude)}``.
    """
    import matplotlib.pyplot as plt
    from matplotlib import gridspec

    if not isinstance(spectra, Mapping) or not spectra:
        raise ValueError("spectra must be a non-empty mapping")
    point_order = list(spectra) if point_order is None else point_order
    if point_sectors is not None and set(point_sectors) != set(spectra):
        raise ValueError("point_sectors and spectra must use the same point names")
    if point_coordinates is not None:
        import cartopy.feature as cfeature
        import cartopy.crs as ccrs

        if set(point_coordinates) != set(spectra):
            raise ValueError(
                "point_coordinates and spectra must use the same point names"
            )
        longitudes = [point_coordinates[name][0] for name in point_order]
        latitudes = [point_coordinates[name][1] for name in point_order]
        fig = plt.figure(figsize=figsize)
        map_ax = fig.add_subplot(
            1, 1, 1, projection=ccrs.PlateCarree(central_longitude=180)
        )
        map_ax.set_extent(
            [min(longitudes)-map_padding, max(longitudes)+map_padding,
             min(latitudes)-map_padding, max(latitudes)+map_padding],
            crs=ccrs.PlateCarree(),
        )
        map_ax.add_feature(
            cfeature.LAND.with_scale('10m'), facecolor='#d8c77c',
            edgecolor='none', zorder=1,
        )
        map_ax.add_feature(
            cfeature.OCEAN.with_scale('10m'), facecolor='#d3dfe3', zorder=0,
        )
        map_ax.set_facecolor('#d3dfe3')
        map_ax.coastlines(
            resolution='10m', color='#8f7b3e', linewidth=0.8,
            antialiased=True, zorder=3,
        )
        gridlines = map_ax.gridlines(
            draw_labels=True, linewidth=0.5, color='0.6', alpha=0.35,
            linestyle='-', zorder=2,
        )
        gridlines.top_labels = False
        gridlines.right_labels = False
        gridlines.xlabel_style = {'size': 10}
        gridlines.ylabel_style = {'size': 10}
        if site_coordinates is not None:
            map_ax.plot(
                *site_coordinates, marker='*', markersize=14,
                markerfacecolor='gold', markeredgecolor='black',
                transform=ccrs.PlateCarree(), zorder=8, label='Site',
            )
        fig.canvas.draw()
        grid = None
    else:
        if gs_rows * gs_cols < len(point_order):
            raise ValueError("gs_rows * gs_cols must fit every spectrum")
        fig = plt.figure(figsize=figsize)
        grid = gridspec.GridSpec(gs_rows, gs_cols)

    last_plot = None
    for index, point_name in enumerate(point_order):
        source = spectra[point_name]
        dataset, close_dataset = _open_spectrum(source)
        try:
            x = np.deg2rad(fix_dir(dataset[dir_n].values))
            y = dataset[freq_n].values
            z = dataset[efth_n].transpose(
                time_n, freq_n, dir_n
            ).values[time_ix] * (np.pi / 180)
            if point_coordinates is None:
                spectrum_ax = fig.add_subplot(grid[index], projection='polar')
            else:
                longitude, latitude = point_coordinates[point_name]
                projected = map_ax.projection.transform_point(
                    longitude, latitude, ccrs.PlateCarree()
                )
                display = map_ax.transData.transform(projected)
                center = fig.transFigure.inverted().transform(display)
                spectrum_ax = fig.add_axes(
                    [center[0]-spectrum_size/2, center[1]-spectrum_size/2,
                     spectrum_size, spectrum_size],
                    projection='polar', zorder=6,
                )
                spectrum_ax.set_facecolor((1, 1, 1, 0.82))
            last_plot = axplot_spectrum(
                spectrum_ax, x, y, z, vmax=vmax, cmap=cmap
            )
            if point_sectors is not None:
                start, end = point_sectors[point_name]
                width = (end - start) % 360
                center = (start + width / 2) % 360
                spectrum_ax.bar(
                    np.deg2rad(center), spectrum_ax.get_ylim()[1],
                    width=np.deg2rad(width), bottom=0,
                    facecolor=(0.0, 0.85, 1.0, 0.10),
                    edgecolor='#00d4ff', linewidth=2.4,
                    align='center', zorder=10,
                )
            spectrum_ax.set_xticklabels([])
            spectrum_ax.set_yticklabels([])
            spectrum_ax.grid(color='white', linewidth=0.6, alpha=0.8)
            spectrum_ax.spines['polar'].set_color('white')
            spectrum_ax.set_title(str(point_name), fontsize=9, pad=4)
        finally:
            if close_dataset:
                dataset.close()

    if (point_coordinates is not None and super_point is not None and
            site_coordinates is not None):
        projected = map_ax.projection.transform_point(
            site_coordinates[0], site_coordinates[1], ccrs.PlateCarree()
        )
        display = map_ax.transData.transform(projected)
        center = fig.transFigure.inverted().transform(display)
        superpoint_ax = fig.add_axes(
            [center[0]-superpoint_size/2, center[1]-superpoint_size/2,
             superpoint_size, superpoint_size],
            projection='polar', zorder=9,
        )
        superpoint_ax.set_facecolor((1, 1, 1, 0.9))
        last_plot = axplot_spectrum(
            superpoint_ax, np.deg2rad(super_point.dir.values),
            super_point.freq.values, super_point.efth.values[time_ix],
            vmax=vmax, cmap=cmap,
        )
        superpoint_ax.set_xticklabels([])
        superpoint_ax.set_yticklabels([])
        superpoint_ax.grid(color='white', linewidth=0.7, alpha=0.85)
        superpoint_ax.spines['polar'].set_color('gold')
        superpoint_ax.spines['polar'].set_linewidth(3)
        superpoint_ax.scatter(
            0, 0, marker='*', s=70, facecolor='gold',
            edgecolor='black', linewidth=0.8, zorder=12,
        )
        superpoint_ax.set_title('SuperPoint', fontsize=10, weight='bold', pad=5)
    if point_coordinates is not None and last_plot is not None:
        colorbar_ax = map_ax.inset_axes([0.34, 0.68, 0.34, 0.025])
        colorbar = fig.colorbar(
            last_plot, cax=colorbar_ax, orientation='horizontal',
        )
        colorbar_ax.set_title('Spectral energy', fontsize=9, pad=3)
        colorbar_ax.tick_params(labelsize=8)
        _format_spectral_colorbar(colorbar)
        map_ax.set_title(
            'Directional spectra at time {0}'.format(time_ix), pad=12
        )
    elif last_plot is not None:
        colorbar = fig.colorbar(
            last_plot, ax=fig.axes, orientation='horizontal',
            shrink=0.35, pad=0.12, aspect=30,
        )
        _format_spectral_colorbar(colorbar)
    return fig


def Plot_stations_spectra(spectra, point_order=None, time_ix=0, gs_rows=1,
                          gs_cols=1, figsize=(10, 10), **variable_names):
    """Legacy name for :func:`plot_input_spectra`."""
    return plot_input_spectra(
        spectra, point_order, time_ix, gs_rows, gs_cols, figsize,
        **variable_names,
    )


def Plot_spectrum(sp, time_ix=0, average=False, ylim=0.49,
                  figsize=(8, 8)):
    """Plot a SuperPoint spectrum at one time or its temporal average."""
    import matplotlib.pyplot as plt

    if average:
        energy = np.nanmean(sp.efth.values, axis=0)
        title = 'Super Point - Mean'
    else:
        energy = sp.efth.values[time_ix]
        title = 'Super Point - time: {0}'.format(sp.time[time_ix].values)
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(1, 1, 1, projection='polar')
    axplot_spectrum(
        ax, np.deg2rad(sp.dir.values), sp.freq.values, energy, ylim=ylim
    )
    ax.set_title(title, fontsize=14)
    colorbar = fig.colorbar(ax.collections[0], ax=ax, orientation='horizontal',
                           shrink=0.7, pad=0.10, aspect=30)
    _format_spectral_colorbar(colorbar)
    return fig


def Plot_seasons_spectra(sp, figsize=(20, 5)):
    """Plot the four seasonal mean SuperPoint spectra in one row."""
    import matplotlib.pyplot as plt
    from matplotlib import gridspec

    seasonal = sp.groupby('time.season').mean()
    x = np.deg2rad(sp.dir.values)
    fig = plt.figure(figsize=figsize)
    grid = gridspec.GridSpec(1, 4)
    last_plot = None
    axes = []
    for index, season in enumerate(('DJF', 'MAM', 'JJA', 'SON')):
        ax = fig.add_subplot(grid[index], projection='polar')
        axes.append(ax)
        last_plot = axplot_spectrum(
            ax, x, seasonal.freq.values,
            seasonal.sel(season=season).efth.values,
        )
        ax.set_title(
            'Season: {0}'.format(season), fontsize=16,
            fontweight='bold', pad=20,
        )
    colorbar = fig.colorbar(
        last_plot, ax=axes, orientation='horizontal',
        shrink=0.35, pad=0.15, aspect=30,
    )
    _format_spectral_colorbar(colorbar)
    return fig


def Plot_bulk_parameters(bulk_params, figsize=(18.5, 9)):
    """Plot significant height, peak period and peak direction."""
    import matplotlib.pyplot as plt
    from matplotlib import gridspec

    fig = plt.figure(figsize=figsize)
    grid = gridspec.GridSpec(3, 1)
    axes = [fig.add_subplot(grid[0])]
    axes.extend(fig.add_subplot(grid[i], sharex=axes[0]) for i in (1, 2))
    styles = (('-', 'darkmagenta', 'Hs'),
              ('-', 'mediumpurple', 'Tp'), ('.', 'navy', 'Dpm'))
    for ax, variable, style in zip(axes, ('hs', 'tp', 'dpm'), styles):
        marker, color, label = style
        ax.plot(
            bulk_params.time, bulk_params[variable], marker,
            markersize=3, color=color, label='Super - Point',
        )
        ax.set_ylabel(label)
    axes[-1].set_xlim(
        bulk_params.time[0].values, bulk_params.time[-1].values
    )
    return fig


def Plot_partitions(stats_part, num_fig=1, figsize=(18.5, 9)):
    """Plot height, period and direction for every spectral partition."""
    import matplotlib.pyplot as plt
    from matplotlib import gridspec

    colors = (
        'navy', 'crimson', 'darkmagenta', 'springgreen', 'purple',
        'lightseagreen', 'indianred', 'orange', 'orchid',
    )
    fig = plt.figure(num_fig, figsize=figsize)
    grid = gridspec.GridSpec(3, 1)
    axes = [fig.add_subplot(grid[0])]
    axes.extend(fig.add_subplot(grid[i], sharex=axes[0]) for i in (1, 2))
    for partition in range(len(stats_part.part)):
        size = 6 if partition in (0, 1) else 3
        data = stats_part.sel(part=partition)
        for ax, variable, label in zip(
                axes, ('hs', 'tp', 'dpm'), ('Hs', 'Tp', 'Dpm')):
            ax.plot(
                stats_part.time, data[variable], '.', markersize=size,
                color=colors[partition % len(colors)],
                label='Partition: {0}'.format(partition),
            )
            ax.set_ylabel(label)
    axes[-1].set_xlim(stats_part.time[0].values, stats_part.time[-1].values)
    axes[0].legend(ncol=3, loc='upper center')
    return fig
