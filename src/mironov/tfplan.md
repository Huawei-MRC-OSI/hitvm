MRC OSI, TensorFlow contribution plan
=====================================

Preface
-------

#### Team members and equipment

MRC OSI is a 5 people team. We have the following equipment:

 * Desktop computer with OpenCL/Vulcan GPU card, already in use.
 * Desktop computer with CUDA GPU card, awaits Ubuntu OS installation by System Administrators.
 * Mobile phone Huawei Honor8 with Mali-T880 GPU, OpenCL/Vulcan, no root access
 
#### Team contribution experience, TVM

MRC OSI team made more than 20 accepted contributions to TVM.
According to out experience, first response to contribution comes from the community in
3 days, and it takes approx. 7 days for community to accept the contribution.

References:
* https://github.com/dmlc/tvm/pulls?q=is%3Apr+is%3Aclosed+author%3Adenis0x0D
* https://github.com/dmlc/tvm/pulls?utf8=%E2%9C%93&q=is%3Apr+is%3Aclosed+author%3Agrwlf

#### Team contribution experience, TensorFlow

So far MRC OSI team sent one contribution, by Denis Khalikov, 24 days ago, no response seen.

Also we know about two simple contributions made by Samuel Siju, Bangalore. One of them were accepted
after 20 days of delay.

References:
 * https://github.com/tensorflow/tensorflow/pull/23877
 * https://github.com/tensorflow/tensorflow/pull/23843
 * https://github.com/tensorflow/tensorflow/pull/23786

#### TensorFlow repository statistics

We studied TF repository and may report the following results:
 1. TensorFlow repository is located at https://github.com/tensorflow/tensorflow and contains sources 
    for both XLA and TensorFlow lite repositories
 2. In 2018, approx. 18000 commits were made to TensorFlow `master` branch (For reference, TVM has approx.
    1500 commits for the same period)
 3. Out of ~18000 commits, ~10000 were made as Pull Requests using GitHub interface. The rest
    of commits were made using Git directly by the Google team.
 4. Out of 10000 PR commits:
    * Approx. 2500 were made by GitHub management robot (email `gardener@tensorflow.org`)
    * Approx. 4400 were made by people who have @google.com suffix in their emails, so they are likely
      also belong to Google, but may be not to Google TensorFlow team.
    * Approx 500 were made from emails of @nvidia.com, @intel.com and @microsoft.com
    * One commit from Huawei were made by Samuel Siju.
    * The rest 2600 commit were accepted from the thirdparty contributors
    
Conclusion:

We see only approx. 2600 of 18000 (14%) commits to TensorFlow came from non-Google employees in 2018. The
rest came from Google employees. The project is driven by Google team almost exclusively, so we should
try establish contacts with them first.

The detailed [report on TensorFlow contribution statistics](http://code.huawei.com/snippets/1158) is available
at code.huawei.com.
 
TensorFlow Contribution plan
----------------------------

So far, MRC OSI team have experience in the following topics:

|Skill|Level of expertize|
|-|-|
| Automatic Differentition | high |
| LLVM code generation | high |
| TVM architecture     | high |
| OpenCL/Vulcan        | medium |
| TVM AutoTuner        | low |
| Quantizatoin         | low |

We are going use available skills to enter the TensorFlow development via exploring XLA compiler as the most familiar to us.
The plan is shown below:

| Project | Contribution point | Date |
|-|-|-|
| TensorFlow | Fix minor issues | Jan 2019 |
| XLA | Fix issues related to running models on Mobile devices | Jan 2019 |
| XLA | Fix issues related to OpenCL/ROCm | Feb 2019 |
| XLA | Investigate of Vulcan support, fixing issues | Feb 2019 |
| TensorFlow/XLA | <ul><li>Investigate of TVM-TF integration issues</li><li>Applying Automatic differentiation in TVM-TF integration</li></ul> | March 2019 |

References
* https://tvm.ai/2018/03/23/nmt-transformer-optimize.html

