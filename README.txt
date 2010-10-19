Doc for each program :

canupo scales... - data1.xyz data2.xyz...
  inputs: scales       # list of scales at which to perform the analysis
                       # The syntax minscale:increment:maxscale is also accepted
                       # Use - to indicate the end of the list of scales
  inputs: dataX.xyz    # raw point clouds to process
  outputs: dataX.msc   # corresponding multiscale parameters for each cloud
TODO: Eliminate points on the border of the scene.
      How ? Simply retain only points within max radius from scene center ?
      Compute scene bounds (how?) and retain points not closer than 1 maxscale from the border
      Better: centering the neighbor cloud is necessary for PCA => compute effective radius
              Then check what given scale is closest the the effective radius
              Points on the very edge of the scene have effective radius approx 1/2 the given scale
              and no stats are available for them at full scale ?
              => remove any point for which stat not available at all scales
                 include isolated points as well.
              BUT then: how to match these points in a real scene ?
              => in fact, if the class def is not polluted too much, better keep the borders
              => trade-off between better global recognition vs better classification for the edge points

annotate data.xyz data.msc annotated_file.xyz [some scales]
  input: data.xyz            # Original data file that was used to compute the multiscale parameters
  input: data.msc            # The multiscale parameters computed by canupo
  input: some scales         # Selected scales at which to perform the annotation
                             # The closest match from the parameter file is selected.
                             # All scales in the parameter file are used if not specified.
  output: annotated_file.xyz # The data with RGB columns corresponding to the local 1D/2D/3D
                             # property at each point. There are 3 such colums per selected scale.
  # Note: can be used on the whole scene as well, the dimensionality characterisation is local


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
  output: some_file.svg   # Multiscale trajectories in the dimensionality feature space


classify features.prm scene.xyz scene_annotated.xyz
  input: features.prm         # Features computed by the make_features program
  input: scene.xyz            # Point cloud to classify/annotate with each class
  input: scene.msc            # Multiscale parameters for the whole scene
  output: scene_annotated.xyz # Output file containing an extra column with the class of each point
  # TODO: Do we really need to compute the multiscale parameters for the whole scene ?
          Can this be restricted to just a few scales ?



