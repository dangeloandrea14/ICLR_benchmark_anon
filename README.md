# A Multi-domain Benchmark for Machine Unlearning in Classification Tasks

## Table of Contents
* [General Information](#general-information)
* [Quick Example](#quick-example)
* [Important Notes](#important-notes)
* [Implementing Custom Unlearners](#custom-unlearners)

## General Information

In this repo there are all the implementation details of our benchmark. 

The code works by simply running main.py with the only argument of the configuration file to use. For instance:

`python main.py cifar100_test_run.jsonc`

All the configuration files are contained in the `config` folder. All the implementation is in the `erasure` folder. Unlearners, in particular, are contained in `erasure/unlearners`.

All datasets (except CelebA) are downloaded automatically if not present.

**Please note!** The configurations were tested in a CUDA environment. Compatibility with other envs is not ensured.

To run all experiments, you can run the appropriate file:

`./reproduce.sh`


## Quick Example

In order to run a quick test run on a small portion of CIFAR100 and a small portion of unlearners, you can run:

`python main.py cifar100_test_run.jsonc`

This will save the result file in a file called cifar100_test.json in the root folder.

The code produces log files in `output/logs`.

This example is meant to be a proof of concept of the benchmark. Since it is run on a small sample of CIFAR100, its results are not reliable.

## Important Notes

1. As mentioned, all experiments were run in a CUDA env to retrieve the max amount of CUDA storage. Compatibility with other environments is not ensured (errors on the data types of the tensors during inference might appear. They are easily fixable).
2. For CelebA, since the source folder is often unavailable, the code expects the folder to be already downloaded.
3. For anonymity reasons, full examples, including a step-by-step guide on extension, will be de-anonymized pending acceptance. 

## Implementing Custom Unlearners

Additional Unlearners can be added by extending the Unlearner class (or its subclass TorchUnlearner). The main methods of Unlearner are \_\_preprocess\_\_, \_\_unlearn\_\_, and \_\_postprocess\_\_. The first one allows the preparation of the model for the unlearning phase. The second is intended to implement the unlearning logic that must be applied to the model. The last method allows to specify any post-process that may be applied to the model. Unlearners can be fully developed using the \_\_unlearn\_\_ method, while the other two methods can be seen as utilities that involve actions that are applied independently of the actual unlearning logic. 
The configuration helps to set all the configuration parameters of the Unlearner easily and of the applied pre-processing step.