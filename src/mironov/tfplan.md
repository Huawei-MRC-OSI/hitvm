MRC OSI, TensorFlow contribution plan
=====================================

Preface
-------

We studied TF repository and may report the following results:
 1. TensorFlow repository is located at https://github.com/tensorflow/tensorflow and contains sources 
    for both XLA and TensorFlow lite repositories
 2. In 2018, approx. 18000 commits were made to its `master` branch (For reference, TVM have approx.
    1500 commits for the same period)
 3. Out of 18000 commits, approx. 10000 were made as Pull Requests using GitHub interface. The rest
    were made using Git directly by the Google team.
 4. Out of 10000 PR commits:
    * Approx. 2500 were made by GitHub robot with email `gardener@tensorflow.org`
    * Approx. 4400 were made by people who have @google.com suffix in their emails, so they are likely
      also belong to Google, but may be not to Google TensorFlow team.
    * Approx 500 were made from emails of @nvidia.com, @intel.com and @microsoft.com
    * One commit from Huawei were made by Samuel Siju.
    * The rest 2600 commit were accepted from the thirdparty contributors
    
As a result, we see only ~2600 of 18000 (14%) commits were accepted from non-Google employees in 2018.
The [detailed report](http://code.huawei.com/snippets/1158) is available at code.huawei.com.
 
TF contribution experience
--------------------------
 
So far MRC OSI attempted to send one contribution, by Denis Khalikov
 * https://github.com/tensorflow/tensorflow/pull/23877
 
Also we know about two contributions made by Samuel Siju, Bangalore
 * https://github.com/tensorflow/tensorflow/pull/23843
 * https://github.com/tensorflow/tensorflow/pull/23786

All changes were trivial, but only one contribution was accepted.
