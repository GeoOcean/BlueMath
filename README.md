# BlueMath

BlueMath is an open-source Python repository designed for the development of both numerical and statistical projects, with the flexibility to operate in a hybrid mode when both approaches are used simultaneously.

<img src="assets/bluemath.png" alt="picture" width="800"/>

At its core lies an open-source package called [BlueMath-tk](https://github.com/GeoOcean/BlueMath_tk), which includes a wide range of tools; from data mining statistical models to wrappers for numerical models. These tools allow users to easily combine both statistical and numerical methods.

BlueMath is designed to be cross-platform and highly efficient when running in the cloud. This enables users to conduct rapid testing in cloud environments like GitHub DevContainers, etc., and run complex metamodels locally or on computing clusters, thanks to generalized parallelization functions.

As shown in Figure 1, this repository provides examples of how to use the core toolkit and how to combine its components to create more complex metamodels (referred to as methods) or climate services.

## Courses

<img src="assets/crono.png" alt="picture" width="800"/>

## Funding

1. Internal project at the GeoOcean group, University of Cantabria.
2. Specific Projects that partially support the development of BlueMath.

<img src="assets/logos.png" alt="picture" width="800"/>

## History

During the last 10 years the GeoOcean research group of University of Cantabria has been developing tools, methods and frameworks for ocean, coastal and hydraulic engineers and scientists.  In terms of the computational environment, and as many research groups, there has been a transition between Matlab and Python.

The Python ecosystem (Jupyter Notebooks, Jupyter Books, Jupyter Hub, Visual Studio Code) offers an open source, transparent and efficient system for running the processes in local computers, private and public HPCs. Besides, the visualization of the notebooks facilitates the interaction with the computer, being nowadays a new standard for teaching, for sharing  and exchanges tools, methods and frameworks.

In September 2020, during COVID pandemic, a virtual workshop leaded by NOAA (PI John Marra) attended by researchers of Spain (University of Cantabria), USA (OSU, SCRIPPS, USGS, FIU, Univ. Hawaii, Duke University), Australia (CSIRO), Fiji (SPC), New Zealand (Uni. of Auckland) was held presenting the idea of a Library of tools, methods and frameworks. This community model library was named BlueMath, aiming for an integration of: (a) existing coastal and ocean engineering practice (methods, data, numerical models); (b) data science (statistics, data mining, artificial intelligence); and (c) climate (seasonality, interannual variability, climate variability, climate change).

BlueMath aims to provide tools (toolkit), methods (hybrid downscaling, statistical downscaling, dynamical downscaling) and climate services (short-term forecast, seasonal forecast, hindcast, probabilistic assessment, climate variability, climate change) for addressing problems at different spatial scales (global, regional, local, high resolution) involving different physical processes (wind, waves, storm surge, tropical cyclones, sea level, morphodynamics,…).

Since the workshop in 2020, more universities has been interested in collaborating in the development of this community model BlueMath: UFSC (Brazil), TU Delft (Netherlands), UCSC (USA), ERDC (USA), IFCA (Spain). The collaboration of these institutions with the GeoOcean group of the University of Cantabria is being carried out through different mechanisms (sharing data bases, computer codes and methodologies, co-op agreements, consultancy projects, research projects, exchange of researchers).

## Working Examples

| Name                | Tested Image Version | Tested Commit                            | Path to Example                                        | Cloud |
| ------------------- | -------------------- | ---------------------------------------- | ------------------------------------------------------ | ----- |
| BinWaves_Duke       | geoocean/rocky8:1.12 | 4b47b92f7d194986a37867a44eeea5500d411076 | climate_services/hindcast/BinWaves_Duke                | ✅     |
| GreenSurge_Tonga    | geoocean/rocky8:1.9  | 0805fa6944049ada65e257d54ee4ae96ad0b5b30 | climate_services/hindcast/GreenSurge_Tonga             | ❌     |
| MUSCLE_Aveiro       | geoocean/rocky8:1.12 | 6eb0f1fc47d9040ca545df41b1013d8b2565b848 | climate_services/probabilistic/MUSCLE_Aveiro           | ✅     |
| HySwash/Veggy       | geoocean/rocky8:1.12 | aa25b0b9cfc7a8ac8311587b1cc6300d75029a10 | methods/hybrid_downscaling/metamodels/HySwash/Veggy    | ✅     |
| HySwash/CHySwash    | geoocean/rocky8:1.9  | 56413fb96147bab9d495af793ccd29cc88fea44f | methods/hybrid_downscaling/metamodels/HySwash/CHySwash | ❌     |
| GreenSurge_Singapur | geoocean/rocky8:1.9  | 0805fa6944049ada65e257d54ee4ae96ad0b5b30 | methods/hybrid_downscaling/additive/GreenSurge         | ❌     |
| SHyTCWaves          | geoocean/rocky8:1.9  | 56413fb96147bab9d495af793ccd29cc88fea44f | methods/hybrid_downscaling/metamodels/SHyTCWaves       | ❌     |

## Usage

To facilitate the usage of all these examples, the [GeoOcean Group](https://geoocean.sci.unican.es/hyweb/) has built a Docker image with all the pre-compiled numercial models, and the required python libraries. This image, and all the information, can be found in [DockerHub](https://hub.docker.com/repository/docker/geoocean/rocky8/general).
