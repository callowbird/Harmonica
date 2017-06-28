# Harmonica
Code repository for the paper [Hyperparameter Optimization: A Spectral Approach](https://arxiv.org/abs/1706.00764) by Elad Hazan, Adam Klivans, Yang Yuan.

## How To Use The Code

(We are using Python 3)

Before using the code, you may need to implement three parts by yourself:
1. "query" method in [samplings.py](https://github.com/callowbird/Harmonica/blob/master/samplings.py). This method has input x (a vector of hyperparameters), and output y (the loss value of this vector). See samplings.py for two simple examples.
2. (Optional) If you want to use hyperband as the base algorithm (see paper for more details), you also need to implement "batch_intermediate_sampling" method in [samplings.py](https://github.com/callowbird/Harmonica/blob/master/samplings.py) for your own application. For the details of this method, please read the [hyperband paper](https://arxiv.org/abs/1603.06560). If you want to use random search as the base algorithm, please comment/uncomment [the line](https://github.com/callowbird/Harmonica/blob/master/main.py#L101) at the last of [main.py](https://github.com/callowbird/Harmonica/blob/master/main.py).
3. (Optional) Harmonica only needs uniform sampling in each stage, therefore it makes sense to implement the sampling process in parallel if you have multiple machines. Please modify "batch_sampling" method in [samplings.py](https://github.com/callowbird/Harmonica/blob/master/samplings.py) accordingly. See [Parallelization Tips](https://github.com/callowbird/Harmonica#parallel-tips) for some suggestions.

You also need to fill [options.txt](https://github.com/callowbird/Harmonica/blob/master/options.txt) to give the names of each hyperparameter. See [options_example.txt](https://github.com/callowbird/Harmonica/blob/master/options_example.txt) for illustration. Do not add extra lines at the end of the file.

Now you may run "python main.py -N 60", if you have 60 hyperparameters. Notice that it also indicates that options.txt contains exactly 60 lines.

Harmonica will run 3 stages by default, and then run a base algorithm for fine tuning. It will output the best answer at the end.

## Two Examples

You may comment/uncomment the two examples in samplings.py, and run "python main.py".

## Algorithm Description

Harmonica extracts important features by a multi-stage learning process. The rough idea is the following (see paper for more details).

    Step 1. Uniformly sample (say) 100 configurations
    Step 2. Extend the feature vector with low degree monomials on hyperparameters
    Step 3. Run Lasso on the extended feature vector, with alpha (weight on l_1 regularization term) equal to (say) 3
    Step 4. Pick the top (say) 5 important monomials, fix them to minimize the sparse linear function that Lasso learned.
    Step 5. Update function f, go back to Step 1.

Keep running this process for (say) 3 stages, and we already fix lots of important variables. Now we can call some base algorithm like Hyperband, Random search (or your favorite hyperparameter tuning algorithm) for fine tuning the remaining variables.

As we can see above, there are a few hyperparameters for Harmonica, like #samples, alpha, #important monomials, #stages. Fortunately, we observe that Harmonica is not very sensitive to those hyperparameters. Usually the default value works well.

## Parallel Tips
It is easy to make Harmonica run in parallel during the sampling process. Here is a simple way for doing it with [pssh](https://linux.die.net/man/1/pssh) in Azure. (EC2 is similar)

    1. Add a few (say 10) virtual machines in the same resource group.
    2. For every machine, set up the corresponding DNS name. Say, hyper001.eastus.cloudapp.azure.com
    3. For simplicity, we rename these virtual machines M1, M2, ... , M10. (You may set up the ssh config file to do so)
    4. Assume the main machine is M1. You may use pssh command to control M1-M10 with just one line:
            pssh -h hosts.txt -t 0 -P -i  'ls'
       Here hosts.txt is a text file with 10 lines. Each line contains the host name for one machine (say, M3, or M5). By running this command, M1-M10 will run 'ls' locally.

Now you are able to run programs on 10 machines simultaneously. How should we make sure everyone is working on different tasks?

First, create a shared filesystem.
* See [how to do it on azure](https://docs.microsoft.com/en-us/azure/storage/storage-how-to-use-files-linux)
* Or you may create your own shared filesystem using [SMB](https://help.ubuntu.com/community/How%20to%20Create%20a%20Network%20Share%20Via%20Samba%20Via%20CLI%20%28Command-line%20interface/Linux%20Terminal%29%20-%20Uncomplicated%2C%20Simple%20and%20Brief%20Way%21)

Once it's done, every machine will have a local directory, say, /shared, which is shared with every other machine.

Now, you may first write a "task.txt" file, which contains, say, 100 tasks. Let's call it T1, T2, ..., T100.

On every machine, you run the program that reads the task.txt file, and understand there are 100 tasks to do. Then the program does the following.


1. It randomly picks (without replacement) a task from the 100 tasks, say T35. Assuming the program is running on M7.
1. It first makes sure that files named "T35M1", "T35M2", ... "T35M10" do not exist in /shared/tags folder. (otherwise someone else is working on it; go back to step 1)
1. It writes a file called "T35M7" in /shared/tags folder, declaring it's working on this task (therefore other machines will not touch T35)
1. After finish T35, it writes "T35Finished" in /shared/tags, and goes back to step 1.

**Remarks**
* If one machine stops unexpectedly, you may restart the program again, and clean the corresponding tags. For example, M7 accidentally stops,
and it was working on T35. Therefore, T35M7 exists in /shared/tags, but T35Finished does not. So M7 should remove /shared/tags/T35M7,
and start working on this task again.
* The program will be able to know it's name by accessing /etc/hostname.
* It is possible that multiple machines will work on the same task, because there is no lock between Step 2 and Step 3. However, it does not happen very often in practice. If you have much more machines than 10 such that it becomes a problem, you may want to use more fancy tools (like map-reduce).


For any questions, please email Yang: yangyuan@cs.cornell.edu.