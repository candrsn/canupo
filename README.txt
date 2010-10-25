Doc for each program :

canupo scales... - data.xyz data_core.xyz data_core.msc
  inputs: scales         # list of scales at which to perform the analysis
                         # The syntax minscale:increment:maxscale is also accepted
                         # Use - to indicate the end of the list of scales
  input: data.xyz        # whole raw point cloud to process
  input: data_core.xyz   # points at which to do the computation. It is not necessary that these
                         # points match entries in data.xyz: This means data_core.xyz need not be
                         # (but can be) a subsampling of data.xyz, a regular grid is OK.
                         # You can also take exactly the same file, or put more core points than
                         # data points, the core points need only lie in the same region as data.
                         # Tip: use core points at least at max_scale distance from the scene
                         # boundaries in order to avoid spurious multi-scale relations
  outputs: data_core.msc # corresponding multiscale parameters at each core point

TODO annotate data.xyz data_core.msc annotated_file.xyz [some scales]
  input: data.xyz            # Original data file that was used to compute the multiscale parameters
  input: data_core.msc       # The multiscale parameters computed by canupo
  input: some scales         # Selected scales at which to perform the annotation
                             # All scales in the parameter file are used if not specified.
  output: annotated_file.xyz # The data with RGB columns corresponding to the local 1D/2D/3D
                             # property at each point. There are 3 such colums per selected scale.
  # Note: the data points take the same characteristics as their nearest neighbor from the core points
  #       defined in the msc file.

density data.msc nsubdiv nametag [some scales]
  input: data.msc              # The multiscale parameters computed by canupo
  input: nsubdiv               # Number of subdivisions on each side of the triangle
  input: nametag               # The base name for the output files
  input: some scales           # Selected scales at which to perform the density plot
                               # All scales in the parameter file are used if not specified.
  output: nametag_scale.svg    # One density plot per selected scale
TODO: allow to merge several msc files in a unique density, in order to have the density
      for one class def as defined in make_features

make_features features.prm data1.msc data2.msc - data3.msc - data4.msc...
  inputs: dataX.msc     # The multiscale parameters for the samples the user wishes to discriminate
                        # Use - separators to indicate each class, one or more samples allowed per class
  output: features.prm  # The resulting parameters for feature extraction and classification of the whole scene  
  # Note: Also displays the characteristic scales that allow each class to be best discriminated against all others
  # Note2: Interactive selection

trajectories some_file.svg features.prm N data1.msc data2.msc - data3.msc - data4.msc...
  input: data.msc         # The multiscale parameters computed by canupo
  input: features.prm     # Features computed by the make_features program
  input: N                # Select the N most representative trajectories for each class for display
                          # This is necessary as one trajectory per point would be unreadable
                          # Use the percentiles. Ex: N=1: use median. N=2: use 33% and 66% percentiles. etc.
                          # percentiles defined as worst/best classified points
                          # => have N representative trajectories of the whole class, no outlier and quite some diversity !
  output: some_file.svg   # Multiscale trajectories in the dimensionality feature space
                          # TODO: find color/representation scheme for each class _and_ for beginning/end of trajectory (scales)
                          # ex: color per class (red, blue, etc) and markers (cross, triangle, star) for scales with legend

classify features.prm scene.xyz scene_core.msc scene_annotated.xyz
  input: features.prm         # Features computed by the make_features program
  input: scene.xyz            # Point cloud to classify/annotate with each class
  input: scene_core.msc       # Multiscale parameters at core points in the scene
                              # This file need only contain the relevant scales for classification
                              # as reported by the make_features program
  output: scene_annotated.xyz # Output file containing an extra column with the class of each point
                              # Scene points are labelled with the class of the nearest core point.


