{
  pkgs ? import <nixpkgs> { },
}:

pkgs.mkShell {
  name = "gravity-sims-env";

  packages = with pkgs.python312Packages; [
    python
    venvShellHook

    # Core Dependencies (from Nixpkgs)
    numpy
    scipy
    matplotlib
    h5py
    torch
    tqdm
    pyqt5
    tensorboard
    pyvista

    # Qt dependencies for visualization
    pkgs.qt5.qtbase
    pkgs.qt5.qtwayland
    pkgs.qt5.qtx11extras

    # System dependencies for compiling/running pip wheels
    pkgs.stdenv.cc.cc.lib
    pkgs.libGL
    pkgs.zlib
    pkgs.glib
  ];

  # Automatically create and enter a virtual environment
  venvDir = "./.venv";

  # Install dependencies not in Nixpkgs (like simpeg, choclo) from requirements.txt
  postVenvCreation = ''
    unset SOURCE_DATE_EPOCH
    pip install -r requirements.txt
  '';

  # Ensure compiled libraries referenced by pip packages can be found
  LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath [
    pkgs.stdenv.cc.cc.lib
    pkgs.libGL
    pkgs.zlib
    pkgs.glib
    pkgs.qt5.qtbase
  ];

  shellHook = ''
    export QT_QPA_PLATFORM_PLUGIN_PATH="${pkgs.qt5.qtbase}/lib/qt-5.15.18/plugins"
    export QT_QPA_PLATFORM="offscreen"
    export PYVISTA_OFF_SCREEN="true"
    export DISPLAY=""
    echo "Qt platform set to offscreen for headless rendering"
  '';
}
