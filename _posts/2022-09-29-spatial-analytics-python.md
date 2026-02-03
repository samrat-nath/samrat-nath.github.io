---
title: 'Introduction to Spatial Analytics in Python'
date: 2022-09-29
permalink: /posts/2022/09/spatial-analytics-python/
tags:
  - python
  - spatial-analysis
  - geodata
  - data-visualization
  - data-science
---

![](https://miro.medium.com/v2/resize:fit:700/1*sDYYo46KvVlsFMbdp5wH_Q.png)

*Image Source:*  [*Pixabay*](https://pixabay.com/illustrations/web-map-flat-design-pin-world-3120321/){:target="_blank"}

[Original Medium Blog Link](https://medium.com/walmartglobaltech/introduction-to-spatial-analytics-in-python-44e92b4af362){:target="_blank"}

Data involving any type of the specific geographical area or location information is called “spatial” data (or “geospatial” data). Geospatial data helps in understanding relationships between geographic attributes and any other metrics data, e.g., how does sales of product vary from urban areas to coastal areas? Geospatial data has various applications, such as

-   Visualizing the area that the data describes
-   Performing trade area analysis
-   Selecting locations for opening new stores for a brand
-   Planning telecommunication/transportation network
-   Risk assessment due to destructive weather, etc.

Obtaining such insights are valuable which makes spatial data skills a great addition to any data scientist’s toolset. In this article, an introductory overview to Python’s spatial analytics ecosystem is provided through some fundamental geospatial operations from ‘geopandas’ package. By the end of this article, readers will learn about

-   Basics of geospatial data format
-   Creating a point geometry from latitude and longitude
-   Creating buffer area around a point for trade analysis
-   Visualizing geospatial data using folium
-   Join operation between two spatial dataframes for intersecting points

**1.**  **Working with Geospatial Data**

**1.1.**  **Vector Data**

Vector data represent geometries in the world. When you open a navigation map, you see vector data. The road network, the buildings, the restaurants, and ATMs are all vectors with their associated attributes. Vector data is simply a collection of discrete locations ((x, y) values) called “vertices” that define one of three shapes:

-   **Point**: a single (x, y) point. Like the location of your house.
-   **Line**: two or more connected (x, y) points. Like a road.
-   **Polygon**: three or more (x, y) points connected and closed. Like a lake, or the border of a country.

*Shapes of different vector data:*
![](https://miro.medium.com/v2/resize:fit:700/1*LdKZXbs3pZ5wlsFXKMrWWA.png)

Vector data is commonly stored in a “shapefile” format. A shapefile is composed of three required files with the same prefix (here, ‘spatial-data’) but different extensions:

-   _spatial-data.shp_: main file that stores records of each shape geometries
-   _spatial-data.shx_: index of how the geometries in the main file relate to one-another
-   _spatial-data.dbf_: attributes of each record

There are other file-types for storing vector data too like geojson. These files can generally be imported into Python using the same methods and packages we use below.

**1.2.**  **Geopandas**

In this article, we will be primarily using open-source python library called  [geopandas](https://geopandas.org/getting_started/introduction.html)  to work with vector data in python. Geopandas extends the  [pandas](https://pandas.pydata.org/pandas-docs/stable/)  capabilities to geospatial data and leverages the capabilities of  [shapely](https://shapely.readthedocs.io/en/stable/manual.html)  to perform geometric operations on spatial data. Geopandas depends on  [fiona](https://fiona.readthedocs.io/en/latest/manual.html)  for file access and matplotlib for plotting. Key datatypes used in geopandas are GeoSeries and GeoDataFrame like Series and DataFrames from Pandas. GeoDataFrames contain geometric column generally called as ‘geometry’. Geometry column contains different geometries like points (latitudes and longitudes), lines, polygons, etc., as shapely objects. Below is schematic view of a GeoDataFrame.

*Data format of a typical geodataframe:*
![](https://miro.medium.com/v2/resize:fit:448/1*36KyfoSncD9306zt8bwgOQ.png)

Next, we will explore some examples of geospatial operations by analyzing a dataset on US fast food restaurants.

**2.**  **US Fast Food Case Study**

In this case study, we’ll be using a dataset from  [Kaggle](https://www.kaggle.com/datasets/datafiniti/fast-food-restaurants?select=FastFoodRestaurants.csv)  which contains information about 10,000 fast-food restaurants in US. For the sake of simplicity, we will only analyze a subset of the dataset. The objective is to locate all the McDonald’s restaurants in New York state and determine how many Burger King restaurants (a competitor of McDonald’s) are in the vicinity of corresponding McDonald’s restaurants.

First, we’ll import the necessary python libraries and load the dataset.

![](https://miro.medium.com/v2/resize:fit:700/1*ihLMUYaaIfHMDwzPPkXb_A.png)

**2.1.**  **Creating Point and Buffer Area**

Note that these datasets are the usual Pandas dataframes. Next, we will convert these into geodataframes by creating a point geometry object from latitude and longitude. The following function uses the Coordinate Reference System (CRS) of WGS84 for converting the dataset into geospataial data. WGS84 is standard for GPS and is made up of a reference ellipsoid, a standard coordinate system, altitude data, and a geoid. The readers are encouraged to check this  [link](https://epsg.io/4326)  to learn more about CRS.

![](https://miro.medium.com/v2/resize:fit:700/1*Oc19CyZS3jL2Ms4ChgEygg.png)

As it can be seen below, a new column ‘Centroid’ has been created which is a Point geometry data type and the class of the dataframe is converted to GeoDataFrame.

![](https://miro.medium.com/v2/resize:fit:700/1*PyqYNVWUZ5ikEoVskEOk8g.png)

Next, we create a buffer area around the centroid points based on a given radius.

![](https://miro.medium.com/v2/resize:fit:700/1*juvvzriFq1n89BytiAUXvQ.png)

As shown below, the GeoDataFrame now consists of another geometry object column named ‘Buffer_Area’ which is a polygon.

![](https://miro.medium.com/v2/resize:fit:700/1*ptTT6t3ZRCmXDKLsK2sfIw.png)

**2.2.**  **Visualizing Geospatial Data**

Now, we are going to visualize both the point (‘Centroid’) and polygon (‘Buffer_Area’) geometry objects on a map using the python library ‘[Folium](https://python-visualization.github.io/folium/)’ and the underlying built-in tile-set  [‘OpenStreetMap’](http://openstreetmap.org/).

![](https://miro.medium.com/v2/resize:fit:700/1*hia8Qm2V1Yt54cVV4CX-VQ.png)


![](https://miro.medium.com/v2/resize:fit:700/1*ZUYKS3eVNVMG-nqCajNOig.png)

*McDonald’s in New York State*


![](https://miro.medium.com/v2/resize:fit:700/1*hgKFizjbcpC4JvmUefn8Gg.png)

*Polygons are drawn around 3 miles from all the McDonald’s*

**2.3.**  **Spatial Join**

In this section, we will find the Burger King restaurant points that fall within corresponding buffer areas of the McDonalds' using Geopandas’ spatial join function. Just like Pandas’ join operation, this one also involves joining two geodataframes both having at least one geometry type variable. However, there are a few additional things that are noteworthy to mention:

-   CRS units for geometry objects from both dataframes should match.
-   The geometry objects in each data frame to be joined should both be named ‘geometry’ or ‘set_geometry’ options should be used to denote the primary geometry object in case there are multiple geometry columns in the dataframe.
-   After performing the inner spatial join operation, only the geometry object data from the left data frame is retained and the other one is discarded.

For details, readers are requested to check the geopandas documentation  [here](https://geopandas.org/docs/user_guide/mergingdata.html). The code below demonstrates how to use spatial_match to find the Burger Kings that are located within 3 miles of McDonald’s in New York.

![](https://miro.medium.com/v2/resize:fit:700/1*u80NdfnbJPUxhNmfd8fqYg.png)

As it can be seen below in the zoomed-in version of the map, the Burger Kings are marked as red points within the blue buffer area of McDonalds'.

![](https://miro.medium.com/v2/resize:fit:695/1*bmu2PALWuxncYorPK57x6g.png)

*Red points indicate the locations of Burger Kings*

The code for this case study can be found in this  [github link](https://github.com/samrat-nath/python-tests/blob/94610d35e2b2d578959f63c324a55b99e7c86965/Misc/spatial_analytics.py).

**3. Conclusion**

Goal of this article was to introduce the concept of geospatial analysis, geopandas and other resourceful open-source python spatial libraries. We have covered common spatial operations like creating geometry points, creating buffer areas, spatial joins, and visualizing geospatial data on maps. There is a lot more that can be done with geospatial analysis like creating the KML files from geospatial data, calculating drive distance between geo points, etc., which we would like to cover in future posts. So stay tuned.

**_Acknowledgment:_** _Thanks to @_[_Sai Manikanta Mukka_](https://medium.com/@saimanikantamukka) _for collaborating on this article_.

**4. References**
- [GeoPandas](https://geopandas.org/en/stable/?source=post_page-----44e92b4af362---------------------------------------)
- [Shapely](https://shapely.readthedocs.io/en/stable/manual.html?source=post_page-----44e92b4af362---------------------------------------)
- [Fiona](https://fiona.readthedocs.io/en/latest/manual.html?source=post_page-----44e92b4af362---------------------------------------)
- [Folium](https://python-visualization.github.io/folium/?source=post_page-----44e92b4af362---------------------------------------)
- [Fast Food Restaurants Across America](https://www.kaggle.com/datasets/datafiniti/fast-food-restaurants?select=FastFoodRestaurants.csv&source=post_page-----44e92b4af362---------------------------------------)
- [OpenStreetMap](http://openstreetmap.org/?source=post_page-----44e92b4af362---------------------------------------)

- [EPSG.io: Coordinate Systems Worldwide](https://epsg.io/?source=post_page-----44e92b4af362---------------------------------------)
