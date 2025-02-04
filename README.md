# BeachSurveys
Programmatic workflows for use with published beach shoreline location data. There are several datasets that exist in the public domain that contain a time-series of beach elevations, but given the variety of formats and conventions used, it is not trivial to use these data for shoreline mapping without considerable pre-processing. This repository is designed to collate and synthesize available datasets, and provide accessible tools to use these data for research into beach/shoreline dynamics, comparison between in situ shorelines and those acquired through remote sensing, and other purposes.

The two principal datasets that have been distilled are:

1. shoreline location mapped onto beach-orthogonal transects, expressed as chainage in meters from each transect's landward origin.
2. beachface slope. The beachface is the portion of the beach between low and high water.



## Sites, and published data:

(If you know of a beach survey dataset that is in the public domain, please raise an Issue and let me know, and I will try to include it)

* Cala Millor Beach, Mallorca, Spain [raw data](https://apps.socib.es/data-catalog/data-products/cala-millor-coastal-station-enl), [paper](https://www.nature.com/articles/s41597-023-02210-2)

* Cardiff/Solana Beaches, California, USA [raw data](https://datadryad.org/stash/dataset/doi:10.5061/dryad.n5qb383), [paper](https://www.nature.com/articles/s41597-019-0167-6)

* Duck Beach, North Carolina, USA [raw data](https://chldata.erdc.dren.mil/thredds/catalog/frf/geomorphology/elevationTransects/survey/data/catalog.html)

* Elwha river delta, Washington, USA [raw data](https://doi.pangaea.de/10.1594/PANGAEA.901486)

* Imperial Beach, California, USA [raw data](https://datadryad.org/stash/dataset/doi:10.5061/dryad.n5qb383), [paper](https://www.nature.com/articles/s41597-019-0167-6)

* Madeira Beach, Florida, USA [raw data](https://coastal.er.usgs.gov/data-release/doi-F7T43S94/)

* Narrabeen Beach, New South Wales, Australia [raw data](http://narrabeen.wrl.unsw.edu.au/), [paper](https://www.nature.com/articles/sdata201624)

* Ocean Beach, California, USA [raw data](https://cmgds.marine.usgs.gov/data-releases/datarelease/10.5066-P13CAWLM/)

* Perranporth Beach, Cornwall, UK [raw data](https://zenodo.org/records/7557390), [paper](https://www.nature.com/articles/s41597-023-02131-0)

* Porsmilin Beach, Brittany, France [raw data](https://portail.indigeo.fr/geonetwork/srv/eng/md.format.html?xsl=doi&uuid=74ecce0a-e650-4c41-9970-97e4602f1cd8), [paper](https://www.nature.com/articles/s41597-022-01170-3)

* Rincon Coast, Puerto Rico, USA [raw data](https://www.sciencebase.gov/catalog/item/61255b87d34e40dd9c03f390)

* Slapton Sands, Devon, UK [raw data](https://zenodo.org/records/7557390), [paper](https://www.nature.com/articles/s41597-023-02131-0)

* Torrey Pines Beach, California, USA [raw data](https://datadryad.org/stash/dataset/doi:10.5061/dryad.n5qb383), [paper](https://www.nature.com/articles/s41597-019-0167-6)

* Truc Vert Beach, Aquitaine, France [raw data](https://osf.io/jftw8/), [paper](https://www.nature.com/articles/s41597-020-00750-5)

* Waikīkī Beach, Hawai‘i, USA [raw data](https://springernature.figshare.com/collections/Three_years_of_weekly_DEMs_aerial_orthomosaics_and_surveyed_shoreline_positions_at_Waik_k_Beach_Hawai_i/6911899/1), [paper](https://www.nature.com/articles/s41597-024-03160-z)

<!-- * Unalakleet, Alaska, USA [raw data](https://coast.noaa.gov/dataviewer/#/lidar/search/-17948942.98192509,9286946.062592342,-17840402.40176014,9363077.342764378) -->


## References
* Bertin, S., Floc’h, F., Le Dantec, N. et al. A long-term dataset of topography and nearshore bathymetry at the macrotidal pocket beach of Porsmilin, France. Sci Data 9, 79 (2022). https://doi.org/10.1038/s41597-022-01170-3

* Brown, J.A., Birchler, J.J., Thompson, D.M., Long, J.W., and Seymour, A.C., 2018, Beach profile data collected from Madeira Beach, FL (ver. 5.0, April 2024): U.S. Geological Survey data release, https://doi.org/10.5066/F7T43S94.

* Castelle, B., Bujan, S., Marieu, V. et al. 16 years of topographic surveys of rip-channelled high-energy meso-macrotidal sandy beach. Sci Data 7, 410 (2020). https://doi.org/10.1038/s41597-020-00750-5

* Fernández-Mora, A., Criado-Sudau, F.F., Gómez-Pujol, L. et al. Ten years of morphodynamic data at a micro-tidal urban beach: Cala Millor (Western Mediterranean Sea). Sci Data 10, 301 (2023). https://doi.org/10.1038/s41597-023-02210-2

* Henderson, R.E., Heslin, J.L., and Himmelstoss, E.A., 2021, Puerto Rico shoreline change — A GIS compilation of shorelines, baselines, intersects, and change rates calculated using the Digital Shoreline Analysis system version 5.1 (ver. 2.0, March 2023): U.S. Geological Survey data release, https://doi.org/10.5066/P9FNRRN0. 

* Hoover, D.J., Snyder, A.G., Barnard, P.L., Hansen, J.E. and J.A. Warrick, 2024, Shoreline data for Ocean Beach, San Francisco, California, 2004 to 2021: U.S. Geological Survey data release, https://doi.org/10.5066/P13CAWLM.

* Ludka, B.C., Guza, R.T., O’Reilly, W.C. et al. Sixteen years of bathymetry and waves at San Diego beaches. Sci Data 6, 161 (2019). https://doi.org/10.1038/s41597-019-0167-6

* McCarroll, R.J., Valiente, N.G., Wiggins, M. et al. Coastal survey data for Perranporth Beach and Start Bay in southwest England (2006–2021). Sci Data 10, 258 (2023). https://doi.org/10.1038/s41597-023-02131-0

* Mikkelsen, A.B., McDonald, K.K., Kalksma, J. et al. Three years of weekly DEMs, aerial orthomosaics and surveyed shoreline positions at Waikīkī Beach, Hawai‘i. Sci Data 11, 324 (2024). https://doi.org/10.1038/s41597-024-03160-z

* Turner, I., Harley, M., Short, A. et al. A multi-decade dataset of monthly beach profile surveys and inshore wave forcing at Narrabeen, Australia. Sci Data 3, 160024 (2016). https://doi.org/10.1038/sdata.2016.24


|  Site | Number of survey dates (since first available Landsat image)  | Survey period  | Average surveys per year  | Median beachface slope  | Offset vertical datum to MSL (m) |
|---|---|---|---|---|---|
|  Cala Millor | 227  | 2011/06/10 to 2020/12/30  | 25  | 0.051  | -0.15 |
|  Cardiff/Solana | 188  | 2007/05/31 to 2016/12/22  | 20  | 0.047  | 0.774 |
|  Duck | 858  | 1984/01/04 to 2023/03/24  | 45  |  0.092 | -0.128 |
|  Elwha | 249  | 2009/04/01 to 2024/01/10  | 16  | 0.15  | 1.165 |
|  Imperial | 95  | 2008/11/14 to 2016/12/12  | 12  |  0.07 | 0.774 |
|  Madeira | 42  | 2016/09/09 to 2023/12/01  | 6  | 0.085  | -0.096 |
|  Narrabeen | 510  | 1976/04/27 to 2019/11/27  | 12  | 0.11  | 0 |
|  Ocean Beach | 213  | 2004/04/07 to 2021/12/06  | 12  | 0.055  | 0.969 |
|  Perranporth | 42  | 2007/12/3 to 2021/01/15  | 3  | 0.028  | 0 |
|  Porsmilin | 280  | 2003/01/08 to 2019/11/28  | 18  | 0.049  | -0.52 |
|  Rincon Coast |   |   |   |   | |
|  Slapton | 247  |  2006/11/21 to 2021/11/6 | 16  | 0.11  | 0 |
|  Torrey Pines | 255  | 2001/02/27 to 2016/12/16  | 17  | 0.04  | 0.774 |
|  Truc Vert | 324  | 2003/09/10 to 2019/12/26  | 20  |  0.05 | 0 |
|  Waikīkī | 129  | 2018/04/12 to 2021/07/22  | 36  | 0.08  | 0|

