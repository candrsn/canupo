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

density nsubdiv nametag [some scales] : [features.prm : ] data.msc [ - data2.msc ...]
  input: nsubdiv               # Number of subdivisions on each side of the triangle
  input: nametag               # The base name for the output files. One density plot is
                               # generated per selected scale, named "nametag_scale.svg"
  input: some scales           # Selected scales at which to perform the density plot
                               # All scales in the parameter file are used if not specified.
  input: data.msc              # The multiscale parameters computed by canupo.
                               # Use - to separate classes. Multiple files per class are allowed.
                               # If no classes are specified (ex: whole scene file) the density is color-coded from blue to red.
                               # If multiple classes are specified the density is coded from light to bright colors with one color per class.
  input: features.prm          # An optional classifier definition file. If it is specified the decision
                               # boundaries at each scale will be displayed in the generated graphs.


features_XXX features.prm [scales] : data1.msc data2.msc - data3.msc - data4.msc...
  output: features.prm  # The resulting parameters for feature extraction and classification of the whole scene
  input:  scales        # Optional, a set of scales to compute the features on.
                        # If this is not specified an automated procedure will find the scales that
                        # best discriminates the given data. You will then be prompted for which
                        # scales to use.
                        # If the scales are specified on the command line there is no interaction and
                        # no automated search.
  inputs: dataX.msc     # The multiscale parameters for the samples the user wishes to discriminate
                        # Use - separators to indicate each class, one or more samples allowed per class
                        # The data file lists start after the : separator on the command line
# Note: XXX stands for the classifier type:
# - least_squares: Basic minimal least squared error hyperplane for each pair of classes
# - linear_svm: Hyperplane separating each pair of classes, but defined so as to lead to maximal margins instead of least squares
# - gaussian_svm: Gaussian kernel SVM, same principle but using the "kernel trick" to get a non-linear mapping in the original space.

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


----
new way:
----

canupo: compute msc params from xyz

suggest_classifier: outfile.svg msc(non label) ... : class1.msc ... - class2.msc ...
    generate svg files
    project on 2 main linear svm dir
    write one default path from gaussian SVM

validate_classifier:  user_modified_svg  [class_num_1  class_num_2]
    produce biclass prm file
    from SVG, use path (predef path if not changed or user-defined)
    if not specified   class_num_1 = 1   and   class_num_2 = 2

combine_classifiers: any number of prm classifiers (incl. multiclass files)
    produce multiclass prm file
    auto from class nums that were specified

classify features.prm scene.xyz scene_core.msc scene_annotated.xyz

