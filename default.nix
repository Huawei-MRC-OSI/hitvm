{ pkgs ?  import ./nixpkgs {}
, stdenv ? pkgs.stdenv
} :

let
  inherit (pkgs) writeText fetchgit fetchgitLocal;
  inherit (builtins) filterSource;
  inherit (pkgs.lib.sources) cleanSourceFilter;

  pypkgs = pkgs.python36Packages;
in

rec {

  tvm-env = stdenv.mkDerivation {
    name = "hitvm-env";

    buildInputs = (with pkgs; [
      cmake
      ncurses
      zlib
      gdb
      universal-ctags
      docker
      gtest
      llvm_6
      clang_6
      openblas
      tig
      protobuf
    ]) ++ (with pypkgs; [
      Keras
      tensorflow
      decorator
      tornado
      ipdb
      nose
      pyqt5
      numpy
      scikitlearn
      matplotlib
      ipython
      jupyter
      scipy
      # mxnet_localssl
      # onnx
      pillow
      pylint
    ]);

    shellHook = ''
      if test -f /etc/myprofile ; then
        . /etc/myprofile
      fi

      if test -f ~/.display ; then
        . ~/.display
      fi

      if test -f dockerenv.sh ; then
        . dockerenv.sh
      fi

      # Fix g++(v7.3): error: unrecognized command line option ‘-stdlib=libstdc++’; did you mean ‘-static-libstdc++’?
      unset NIX_CXXSTDLIB_LINK NIX_TARGET_CXXSTDLIB_LINK
    '';
  };

}.tvm-env
