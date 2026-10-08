# Architecture

## Internal Dependencies

- `latent_space_visualizer.py` -> `dirac_crystal2.py`

## External Imports

- `dirac_crystal2.py` -> abc, argparse, collections, copy, dataclasses, datetime, json, logging, math, numpy, os, time, torch, torch.nn, torch.nn.functional, torch.optim, torch.utils.data, typing, warnings
- `dirac_crystallography_suite.py` -> abc, argparse, collections, copy, dataclasses, datetime, glob, json, logging, math, matplotlib, matplotlib.gridspec, matplotlib.pyplot, numpy, os, pathlib, re, scipy, scipy.linalg, scipy.optimize, scipy.sparse, scipy.sparse.linalg, scipy.stats, seaborn, sklearn.decomposition, time, torch, torch.nn, torch.nn.functional, torch.optim, torch.utils.data, traceback, typing, warnings
- `latent_space_visualizer.py` -> PyQt5.QtCore, PyQt5.QtGui, PyQt5.QtWidgets, collections, csv, dataclasses, datetime, matplotlib, matplotlib.backends.backend_qt5agg, matplotlib.colors, matplotlib.figure, matplotlib.pyplot, mpl_toolkits.mplot3d, numpy, os, pathlib, sklearn.decomposition, sklearn.preprocessing, sys, threading, time, torch, torch.nn, torch.utils.data, traceback, typing
- `lidar_interactive_viewer.py` -> argparse, csv, json, numpy, pathlib, sys, typing
- `relativistic_hydrogen.py` -> abc, dataclasses, glob, json, logging, math, matplotlib, matplotlib.colors, matplotlib.pyplot, numpy, os, scipy, scipy.special, sys, torch, torch.nn, torch.nn.functional, traceback, typing, warnings
- `visualize_lidar_csv2.py` -> argparse, csv, matplotlib.pyplot, mpl_toolkits.mplot3d, numpy, pathlib, sys
- `weight_3d_standard.py` -> argparse, csv, json, numpy, pathlib, sklearn.decomposition, sklearn.manifold, sklearn.preprocessing, sys, torch, typing
- `weight_space_lidar.py` -> __future__, abc, argparse, csv, dataclasses, datetime, json, laspy, logging, math, matplotlib.pyplot, mpl_toolkits.mplot3d, numpy, os, pathlib, re, scipy, scipy.linalg, scipy.spatial.distance, sklearn.decomposition, sklearn.manifold, sys, torch, torch.nn, typing, warnings
