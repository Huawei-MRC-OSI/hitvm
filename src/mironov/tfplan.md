MRC OSI, TensorFlow contribution plan
=====================================

Preface
-------

### Team members and equipment

MRC OSI is a 5 people team. We have the following equipment:

 * Desktop computer with OpenCL/Vulcan GPU card, already in use.
 * Desktop computer with CUDA GPU card, awaits Ubuntu OS installation by System Administrators.
 * Mobile phone Huawei Honor8 with Mali-T880 GPU, OpenCL/Vulcan, no root access
 
### Team contribution experience, TVM

MRC OSI team made more than 20 accepted contributions to TVM.
According to out experience, after contribution were made to TVM, first response from the community comes in
3 days, and it takes approx. 7 days for community to accept the contribution.

References:
* https://github.com/dmlc/tvm/pulls?q=is%3Apr+is%3Aclosed+author%3Adenis0x0D
* https://github.com/dmlc/tvm/pulls?utf8=%E2%9C%93&q=is%3Apr+is%3Aclosed+author%3Agrwlf

### Team contribution experience, TensorFlow

So far MRC OSI team sent one contribution, by Denis Khalikov, 24 days ago, no response seen.

Also we know about two simple contributions made by Samuel Siju, Bangalore. One of them were accepted
after 20 days of delay.

References:
 * https://github.com/tensorflow/tensorflow/pull/23877
 * https://github.com/tensorflow/tensorflow/pull/23843
 * https://github.com/tensorflow/tensorflow/pull/23786

### TensorFlow repository statistics

We studied TF repository and may report the following results:
 1. TensorFlow repository is located at https://github.com/tensorflow/tensorflow and contains sources 
    for both XLA and TensorFlow lite repositories
 2. In 2018, approx. 18000 commits were made to its `master` branch (For reference, TVM have approx.
    1500 commits for the same period)
 3. Out of 18000 commits, approx. 10000 were made as Pull Requests using GitHub interface. The rest
    were made using Git directly by the Google team.
 4. Out of 10000 PR commits:
    * Approx. 2500 were made by GitHub management robot (email `gardener@tensorflow.org`)
    * Approx. 4400 were made by people who have @google.com suffix in their emails, so they are likely
      also belong to Google, but may be not to Google TensorFlow team.
    * Approx 500 were made from emails of @nvidia.com, @intel.com and @microsoft.com
    * One commit from Huawei were made by Samuel Siju.
    * The rest 2600 commit were accepted from the thirdparty contributors
    
As a result, we see only ~2600 of 18000 (14%) commits were accepted from non-Google employees in 2018.
The [detailed report](http://code.huawei.com/snippets/1158) is available at code.huawei.com.
 


