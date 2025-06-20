Future Ideas:

## 1.) Optimize the remove useless attributes speed, and the order of attribute removal!
- We have seen great results in the simplifications they can increase accuracy, and massively reduce the clause count.
    Right now however, we are limited because the RemoveUselessAttributes (R2A in the paper) function in simplifications is very slow.
    There are a few ways to make this better. The simplest is probably to make a second function. Right now, the version we have
    is able to handle disjunctions, to do this, it has to encode the count of each attribute into the mins and maxes arrays.
    Creating a version which just assumes each attribute has one clause, a min and max value, would run much faster.
    If the memory accessing is coalesced, like we have in the merging function, this should be much faster (just transposing a matrix basically, it's not very complicated to coalesce memory access in CUDA).

- To do this faster, just dont encode the rule count in the mins and maxes arrays anymore. and then, we would generate HBs initially, simplify with this new version,
    and if we simply a SECOND time you could run the one which is implemented.

## 2.) Refactoring of merging / increasing speed of generation with new algorithmic changes.
If future speed becomes a larger issue in HB generation, refactoring the merging would be useful. It may be more powerful to actually implement the
    mergerNotInCuda function into cuda. This function does the merging exactly the same way, but the way in which we are checking points is more efficient, because you
    dont check the entire dataset. This means we would have to do some tracking and synchronization within a block in some way of which points are inside a block.
    - avoid having to check the entire dataset, use the data in sorted columns and track which points are within each bound. that way we only check the points which are within
    some of our intervals. You'd have to find a creative way to replicate the set logic in CUDA, maybe just a bit masking thing. it can be done.
    - if we made this faster checking, you could apply that to the remove attributes function as well, to make the simplification much faster as well.
        * the faster algorithm is implemented in IntervalHyper::mergerNotInCuda. But right now we don't have that set up to run on GPU, so the GPU version has been faster in testing. The logic is identical,
        we just used set based checking based on indexes instead of checking every point's value for every attribute.

## 3.) Disjunctions need implemented. 
We set up everything to be compatible with disjunctions, hence why all the Hbs are stored as vector<vector<float>>. Each attribute gets its own vector.
    Disjunctions are implemented in DV, and the logic could be shoplifted over. Another interesting tweak however, instead of treating being in any interval as "inside the HB", we could change the logic to be
    you have to be inside min1 all the way through, and then max1. So it would essentially be like a fork in the road. Right now, you could be in min1-max1, then go up to min2-max2. We don't see a lot of disjunctions
    formed in this way, because it is so permissive. Doing it the other way, where you have to use mins1 and maxes1, OR (exclusively) mins2-maxes2 would allow us to combine blocks which are sharing intervals. Any attribute
    which has the same interval, or one which is mergeable, would be combined. This would probably reduce clauses nicely.
    - in theory disjunctions (in this exclusive or fashion) may not increase accuracy, but would reduce complexity.

To complete this:
- Find the simplifyToDisjunctive algorithm within DV2.0
- Translate the Java version that you find into C++
- Ensure that ALL uses of ".minimums" or ".maximums" assume that the disjunctions are now being used. For example,
  the current 'saveBasicHBsToBinary' in 'data_utilities/DataUtil.cpp' would need expanded upon, since it is "basic HBs" meaning there is currently NOT
  support for importing/exporting disjunctive unit HBs. 


## 4.) Using Validation Data to Improve HB Voting. 
Validation data and more complex voting should increase accuracy if we can figure out a good way of doing that. We have experimented with using a validation portion of the training data
    to determine which HBs are better predictors, and then using those HBs with a heavier vote when the time comes to vote on points.
    - validation data can be used to find the best removal counts as well. just as we are using the findBestParameters function in Host.cu, we could use a validation split
    of the original training data to find best parameters for the whole dataset.

The precision weighted HBs have proved beneficial in some cases, for example WBC, we were able to boost the accuracy. However, we should be careful of
when this is applied. For example, leaning towards Malignant classification for Cancer datasets, since we would NEVER want to tell someone that they are healthy
if they are actually on the borderline (50 HBs vs 50 HBs).

## 5.) Fallback classification ideas. (When a point from testing falls into none of the HBs, so we must decline or find some way to produce a prediction that remains interpretable)

Experiment with fallback classifiers, in the ClassificationTesting prediction function, we could just add another case to the switch, and make a new predictor function.
    This is made like this because we've tried a million different fallback classifiers. Something completely different from KNN could work well, maybe also using something lke k-nearest-HBs, similar to lincoln's version.
    - use the mergableKNN. optimizing this may show better accuracy. It is implemented as Knn::mergableKNN right now, but so computationally expensive you can't run it well with larger sets.
        * may need to be ported to CUDA perhaps. There are other ways to try this as well. But the idea is there, determine which HB to go to based on whether or not you can expand it's bounds to capture a new point.
    - a version of KNN which only uses points which are INSIDE HBs. So we would do something like count which HB has the most near neighbors to an unclassified point.
        * either count how many neighbors, or see which block has most neighbors within ETS thresholds somehow.

## 6.) Simple HB visualization -- CRUCIAL: MUST NOT IMPACT ABILITY TO RUN WITHOUT VISUALIZATION ON CLOUD COMPUTERS SUCH AS THE CWU CUDA CLUSTER
It would be simpler, though not mission critical, to make a python script or something simple which is either an interactive gui, or just plots the HBs, so they can be visualized.
    This is possible in DV, we have done it, but it is a pain to fire up DV, and on some machines DV itself is hard to get running.

## 7.) Another way to generate HBs which could be interesting. 
This may generate many times faster. Consider the sorted columns as if we have ran the separate by attributes function. 
    - generate our seed HBs as we would normally with any of the interval hyper algorithms. 
- a.) take a pure HB. in each column (attribute) keep a tracker of which points are inside of at least one interval.
- b.) once we have made this list using all columns of the dataset, we may have a list like p3, p6, p8, p10. we know that we can expand our bounds up to the point where we would let one of those points in.
            - for example, we may find point 10 in our bounds for attribute 3. we may find that we can expand our x1 from x1min[0.4] to x1min[0.3] before we let in other points.
        * this makes it a bit easier to find a "more optimal" expansion. and then we may run the merging algorithm after this portion too even. 

## 8.) FIX THE IMPORTING OF TEST DATA BRICKING IF THERE HAS ALREADY BEEN A SET IMPORTED 
Essentially, if you import Iris.csv, then try to change to like diabetes.csv, it will crash. This is likely due to 
mismatch in the training data vector size, so its just a simple out of bounds error. However, you will also need to reset the 
things such as class_map_int

## 9.) Find methods that allow us to reduce the number of training points without hurting the classification accuracy.
Essentially some method SIMILAR to Condensed Nearest Neighbors, but adapted to respect the HB structure. CNN currently does 
not do well due to the fact that it picks up points that are on the boundaries, which are important for methods like KNN and even very helpful.
However, when we do this with HBs it can cause ONLY the boundary areas to be represented, meaning we may not make any blocks if they are all noisy regions. 