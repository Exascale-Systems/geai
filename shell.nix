{
  pkgs ? import <nixpkgs> { },
}:

pkgs.mkShell {
  name = "gravity-sims-env";

  buildInputs = with pkgs; [
    python312
    python312Packages.pip
    python312Packages.virtualenv

    # System dependencies for compiling/running pip wheels
    stdenv.cc.cc.lib
    libGL
    zlib
    glib
    gcc
    pkg-config

    # Qt dependencies for visualization
    qt5.qtbase
    qt5.qtwayland
    qt5.qtx11extras
  ];

  shellHook = ''
    export QT_QPA_PLATFORM_PLUGIN_PATH="${pkgs.qt5.qtbase}/lib/qt-5.15.18/plugins"
    export QT_QPA_PLATFORM="offscreen"
    export PYVISTA_OFF_SCREEN="true"
    export DISPLAY=""

    # Create venv if it doesn't exist
    if [ ! -d ".venv" ]; then
      echo "Creating virtual environment..."
      python3 -m venv .venv
    fi

    source .venv/bin/activate
    echo "Virtual environment activated"

    # Install requirements if venv was just created
    if [ ! -f ".venv/.installed" ]; then
      echo "Installing requirements..."
      pip install --upgrade pip
      pip install -r requirements.txt
      touch .venv/.installed
    fi
  '';
}
