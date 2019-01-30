#!/bin/sh

if ! test -d "$CWD" ; then
  echo "CWD is not set. Did you source dockerenv.sh?"
  exit 1
fi

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

