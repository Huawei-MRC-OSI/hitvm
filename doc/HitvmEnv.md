TVM/NNVM environment setup
==========================

This document describes the working process regarding TVM/NNVM project arranged
in Moscow Research center.

Overall procedure
-----------------

  1. Login to `ws` machine (contact Sergey Mironov mWX579795 in order to get user
     account)

        $ ssh 10.122.85.37

  2. Clone the current repository recursively

        $ git clone --recursive https://http://code.huawei.com/mrc-cbg-opensource/hitvm
        $ cd hitvm

  3. Run the docker with `./rundocker.sh` script
  
  4. Within docker, source `. dockerenv.sh` script which define builder help functions.
  
  5. Within docker, build the project (usually `dmake -j10`) and run ipython for
     futher development.

Below we describe build procedure and other development tasks in more details.

Building the TVM/NNVM
=====================

Obtaining TVM repository
------------------------

**Solutions to typical problems may be found
[here](http://code.huawei.com/mrc-cbg-opensource/hitvm-internal/tree/master/mironov/md/README.md)**

Download upstream TVM repository and save to to some `/path/to/tvm` folder.

Running Docker container
------------------------

This environment provides [rundocler.sh](../rundocler.sh) script which can be
used to run interactive docker session for local development.

First, we have to setup `./src/$USER/tvm` link to point to valid TVM repo. This
link will be used to execute Docker rules, defined in TVM project

    $ ln -s ./src/$USER/tvm /path/to/tvm

Next, execute the following script to build and run the docker:

    $ ./rundocker.sh --map-sockets

The `Dockerfile.dev` will be used for building the docker image. One may
adjust/modify/duplicate it as needed.

Finally, source dockerenv.sh to access useful shell functions `dmake`, `dclean`,
`dtest` and others:

                    # inside docker container #
    
    $ . dockerenv.sh
    $ type dmake    # Print dmake's source
    $ dmake -j5
    

Running TVM tasks
=================

Running Python programs using TVM/NNVM
--------------------------------------
*TODO: Check the statement below. Maybe we already use `ipython`*
Same as native mode, but one should use `ipython3` instead of `ipython`.

    (docker) $ ipython3
    >> from reduction import *
    >> lesson1()


Running C++ programs using TVM/NNVM
-----------------------------------

Should be the same as for native mode.


Running Tensorboard within docker
---------------------------------

Tensorboard is a web-application allowing users to examine the structure and
performance metrics of Tensorflow models.

`rundocker.sh` script maps port equal to `6000 + USER_ID -
1000` to port 6006 of the Host (if the `--map-sockets` option is passed).
The exact values should be printed during container startup:

    *****************************
    Your Jupyter port: XXXX
    Your Tensorboard port: YYYY
    *****************************

Tensorboard may then be run with a helper (one may invoke `type dtensorboard`
to review its code):

    (docker) $ dtensorboard &

Tensorboard will run in a background. After that, browser may be used to
connect to Tensorboard webserver:

    $ chromium http://10.122.85.190:YYYY

The connection should be made to port YYYY of the Host machine, the traffic
should be redirected to port 8008 of the Docker container

`dtensorboard` creates and selects `./_logs` as a directory to search for logs.
Tensorflow models should be set up accordingly to save the logs into this
directory.


Running jupyter-notebook within docker
--------------------------------------

Jupyter notebook may be started by typing `djupyter` command, defined by
`dockerenv.sh` script.

    (docker) $ djupyter

After that, a browser may be started from the remote workstation.

    $ chromium http://10.122.85.190:XXXX

Mind the exact value of XXXX from the output of `./rundocker.sh` script. The
connection should be made to port XXXX of the Host machine, which redirects the
traffic to port 8888 of the Docker


Obtaining core dumps in Docker
------------------------------

Docker containers use host's core patterns file `/proc/sys/kernel/core_pattern`
but don't have `apport` installed, so the default setup doesn't do anything
useful on segmentation fault. If you have a segmentation fault which doesn't
produce a core file: 

 1. Run `cat /proc/sys/kernel/core_pattern` (on the host or in the container,
    it doesn't matter. 
 2. If it contains `apport`, do (on the host!)

        $ echo '/tmp/core.%t.%e.%p' | sudo tee /proc/sys/kernel/core_pattern

 3. Somteimtes one has to execut `ulimit -c unlimited` from the terminal where 
    the core dump should be saved.

Reference: https://le.qun.ch/en/blog/core-dump-file-in-docker/

Debugging python scripts
------------------------

 1. Recently we added `--debug` argument to `dmake`. It enables building TVM with
    debug information
        
        $ dmake --debug -j20

 2. Run your faulty script like this:
 
        $ gdb  --args `which python3.6` lstm2.py

AddressSanitizer (aka ASan) for TVM
-----------------------------------
1. Instrument libtvm with ASan.
    
    ```bash
    $ export CFLAGS='-fsanitize=address -g -fno-omit-frame-pointer'
    $ export CXXFLAGS='-fsanitize=address -g -fno-omit-frame-pointer'
    $ cmake -DUSE_LLVM=ON path/to/tvm
    $ make -j32
    ```

2. Preload libasan. 

    ```bash 
    $ export LD_PRELOAD=/path/to/libasan.so
    ```
    
3. Run your program.
    
    ```bash
    $ python3 tvm_model.py
    ```
    
    Official documentation 
    `https://github.com/google/sanitizers/wiki/AddressSanitizer`
 
   
TVM profiling
--------------
1. Callgrind from Valgrind.
 
    To get a better callgraph rebuild libtvm.so with `-00` in other way
    your compiler can inline some functions.

    ```bash
    $valgrind --tool=callgrind python3 lstm.py
    $kcachegrind callgrind.out.number
    ```
    
    Official documentation http://valgrind.org/docs/manual/cl-manual.html
    


