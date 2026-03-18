{
  pkgs ? import <nixpkgs> {
    config = {
      allowUnfree = true;
      cudaSupport = true;
    };
  },
}:

let
  # Python with torch from nixpkgs (this has working CUDA)
  pythonWithTorch = pkgs.python312.withPackages (
    ps: with ps; [
      torchWithCuda
      numpy
      scipy
      matplotlib
      h5py
      tqdm
      pyqt5
      tensorboard
      pyvista
      pip
      dvc
    ]
  );
in

pkgs.mkShell {
  name = "gravity-sims-env";

  packages = [
    pythonWithTorch
    pkgs.uv # Use uv for faster pip installs

    # System dependencies
    pkgs.stdenv.cc.cc.lib
    pkgs.libGL
    pkgs.zlib
    pkgs.glib

    # Qt dependencies
    pkgs.qt5.qtbase
    pkgs.qt5.qtwayland
    pkgs.qt5.qtx11extras

    # NVIDIA support
    pkgs.linuxPackages.nvidia_x11
  ];

  shellHook = ''
    export QT_QPA_PLATFORM_PLUGIN_PATH="${pkgs.qt5.qtbase}/lib/qt-5.15.18/plugins"

    # Create venv with system site packages to access nixpkgs torch
    if [ ! -d ".venv" ]; then
      echo "Creating virtual environment..."
      ${pythonWithTorch}/bin/python -m venv --system-site-packages .venv
    fi

    source .venv/bin/activate

    # Export PYTHONPATH so subprocesses (like DVC) can find packages from both nixpkgs and venv
    export PYTHONPATH=".venv/lib/python3.12/site-packages:${pythonWithTorch}/${pythonWithTorch.sitePackages}:$PYTHONPATH"

    # Install packages not in nixpkgs
    if ! python -c "import discretize" 2>/dev/null; then
      echo "Installing discretize, simpeg, choclo..."
      pip install discretize simpeg choclo
    fi

    # Install project in editable mode
    if ! python -c "import src" 2>/dev/null; then
      echo "Installing project in editable mode..."
      pip install -e .
    fi

    echo ""
    echo "Environment ready! Testing CUDA..."
    python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
  '';
}
