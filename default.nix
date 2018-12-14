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


  mktags = pkgs.writeShellScriptBin "mktags" ''
    (
    cd $CWD
    find src tvm -name '*cc' -or -name '*hpp' -or -name '*h' -or -name '*\.c' -or -name '*cpp' | \
      ctags -L - --excmd=number --c++-kinds=+p --fields=+iaS --extras=+q --language-force=C++

    while test -n "$1" ; do
      case "$1" in
        py)
          find tvm src -name '*py' | ctags --excmd=number --append -L -
          ;;
        tf)
          echo "Building Tensorflow tags" >&2
          find -L _tags/tensorflow -name '*py' | ctags --append -L -
          cat tags | grep -v -w import | grep -v -w _io_ops | grep -v -w 'ops\.' > tags.2
          mv tags.2 tags
          find -L _tags/tensorflow -name '*cc' -or -name '*h' | ctags --append --language-force=C++ -L -
          ;;
        *)
          echo "Unknown tag task: $1" >&2
          ;;
      esac
      shift
    done
    )
  '';


  tvm-env = stdenv.mkDerivation {
    name = "tvm-env";

    buildInputs = (with pkgs; [
      cmake
      ncurses
      zlib
      mktags
      gdb
      universal-ctags
      docker
      gtest
      llvm_6
      clang_6
      openblas
      tig
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
