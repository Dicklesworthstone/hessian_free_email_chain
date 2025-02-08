# A Series of Emails On Hessian Free Methods for Deep Learning:

**From:** Jeffrey Emanuel  
**To:** James Martens  
**Date:** December 14, 2012 2:31 AM

Hi James,

I wanted to thank you for your amazing Matlab code for training neural nets. I had trouble understanding the code the first time I looked at it, so I instead focused on pre-training with DBNs (which does work pretty well). But my mind was blown tonight when I saw that your code got me results in less than 30 minutes that I couldn't get with 15 hours spent on CD and backprop. I am really excited about training very large, very deep nets with it.

Anyway, in the process of understanding the code I made some changes, and hopefully, improvements (see attached). In particular, I replaced every instance of repmat with bsxfun (the original lines are all commented out). This uses less memory because it basically acts like a pointer instead of explicitly allocating the space like repmat does. I think it's also faster, and has built in support under Jacket. I also made the input parameter names longer and more descriptive, and changed the other parts to make it consistent with the new names. There are a couple other things as well.

I was inspired by the recent google paper about asynchronous SGD and was trying to think about a scheme for adapting this code to work that way. I was thinking that you could store the file with the state of the network on dropbox, and then have a simple publish/subscribe interface for asynchronously updating the network state with multiple machines.

Apparently these processes are pretty robust and can work well even if there is no theoretical basis to support it (i.e., each network replica has a slightly out of date version of the weights/biases because it has changed while it was in the middle of an epoch). If I can make anything work I'll be happy to send it over to you. Thanks again for this great work though-- hopefully you will get some nice press for it (like Hinton is now deservedly getting in the NYT) when this AI thing really blows up in the next couple years.

PS: Should I try to move over to using recurrent NNs? Are they generally more powerful and expressive if you can train them? It would be great to have a Matlab version of that code-- there are still a lot of people tied to that environment.

-Jeff

---

**From:** James Martens  
**To:** Jeffrey Emanuel  
**Date:** December 19, 2012 5:59 PM

Hi Jeffrey,

Sorry for the late reply. I've been exceptionally busy lately.

> I wanted to thank you for your amazing Matlab code for training neural nets...

Thanks.

Did you find that the use of bsxfun made a significant speed improvement when using Jacket?

> I was inspired by the recent google paper about asynchronous SGD and was trying to think about a scheme for adapting this code to work that way...

I'm slightly sceptical about the advantage of computing updates with multiple independent optimization runs and then combining them linearly. I wonder if there is a distinct advantage over just computing a bunch of gradients on different datasets and then combining them centrally.

But if it works, it works. I would be interested to know if you have any success with it. I would suggest making a simple proof-of-concept implementation of it before you develop something more elaborate.

> Apparently these processes are pretty robust and can work well even if there is no theoretical basis to support it (i.e., each network replica has a slightly out of date version of the weights/biases because it has changed while it was in the middle of an epoch)...
 
Thanks, and good luck. I'll be interested to know how it turns out.

Cheers,
James

---

**From:** Jeffrey Emanuel  
**To:** James Martens  
**Date:** December 19, 2012 6:33 PM

Thanks for replying, and I certainly understand that you are very busy. 

I haven't done careful benchmarking of bsxfun, but it helped me avoid running out of memory on the GPU (often, the line that cause the memory to run out was one with a "repmat" in it). I imagine it is slightly faster simply because you don't have to wait for it to allocate the additional memory. It is amazing to me how much better these cards have gotten lately. If you are using an older Tesla (like the 2070), you might want to take a look at the newest cards. I just ordered a gamer card last night (http://www.newegg.com/Product/Product.aspx?Item=N82E16814130781 ) that is something like 4 or 5 times faster than my old Tesla, for half the price!

I hear you on being skeptical about the asynchronous updates. I haven't delved into that but will let you know if it turns out to be useful.

I was a bit embarrassed after sending you my modified code to realize that I had misinterpreted the "drop" variable to have something to do with dropout. I have since spent much more time with the code and understand it much better. The main issue I was having was that the trained model would start over-fitting pretty quickly on the validation data, even using an L2 prior on the weights. I implemented early stopping, but that seems to be a very crude method. So I investigated trying to do something like dropout with your code. Since your method is deterministic, I'm not sure if injecting stochasticity with dropout is even sensible. Still, I tried adding it in two ways. 

The first is like the method described in the Hinton paper, but instead of randomly selecting half of  all the hidden units, I only sample from the non-zero ones, since you are using sparse initialization. That method seems to be of questionable value based on my tests, although I'm still not 100% sure. The other method I tried is to randomly pick a percentage of the weights and to then randomly perturb them by either dilating or contracting by a small factor (e.g., multiplying by 0.99 or 1.01). This is not done to the actual weights to avoid excessive "corruption"; they are instead copied to a temporary variable right after the " rdEdy{ii+1} = [];" line of the "computeGV" function and then the random perturbations are applied to that variable. The perturbed weights are then used to compute " rdEdy{ii} ". I also did something similar to the inputs themselves as they are formed into mini-batches, so that different perturbations will be applied for every epoch.

On top of that, I pulled out several hard-coded values in your code and made them parameters of the function: the number of incoming connections per unit, the lower bound for the value of rho to apply the damping heuristic, the initial lambda, and the maxiters. I also made as parameters of the function the various quantities described earlier for my attempt at introducing dropout. Then I used CMA-ES ( http://www.lri.fr/~hansen/cmaesintro.html ) to explore the space of hyper-parameters, where the objective function was based on the model's validation performance. This required making a simple wrapper function to your code to plug in the CMA-ES search vector to the relevant parameters, but it wasn't too hard. I've only been running the search for a couple days, but I can already see that at least in some cases, randomly perturbing the weights seems to help. I have also found that I could get better results (depending on the network architecture) with a higher number of incoming connections. The best run I've had so far used a weight cost of 3e-05, lower bound for damping heuristic of 0.2875 (the upper bound is always set to 1 - lower_bound), number of incoming connections of 12, initial lambda of 23, and randomly perturbed 76% of the weights.

I think this method of programmatically exploring the set of hyper-parameters is a promising one. It allows you to try crazy things that don't have much theoretical support, and if they end up helping a lot, you can try to go back and understand what is going on and try to formalize things. But the main benefit is that you can keep your hardware busy while you sleep without having to intervene, and CMA-ES will do a really nice job intelligently trying different things. To make it work well I had to also add various stopping criteria so it would kill bad runs without wasting a lot of time. So, for example, if I get more than a 2 consecutive rho = -Inf in the first first 5 epochs, I will kill the run. Or if there are too many rejected steps, I will kill the run. Mostly I will kill the runs if my validation performance (measure by AUPRC) is too low by the Nth epoch, or if the median improvement percentage of the AUPRC is too low by the Nth epoch (these constraints are specified with vector so I can gradually raise the bar as it finishes more epochs).

Anyway, hopefully you find these explorations at least moderately interesting.

-Jeff

---

**From:** James Martens  
**To:** Jeffrey Emanuel  
**Date:** December 21, 2012 12:29 AM

> Thanks for replying, and I certainly understand that you are very busy. I haven't done careful benchmarking of bsxfun, but it helped me avoid running out of memory on the GPU (often, the line that cause the memory to run out was one with a "repmat" in it). I imagine it is slightly faster simply because you don't have to wait for it to allocate the additional...

Interesting.  I may give it a try then.

> memory. It is amazing to me how much better these cards have gotten lately. If you are using an older Tesla (like the 2070), you might want to take a look at the newest cards. I just ordered a gamer card last night (http://www.newegg.com/Product/Product.aspx?Item=N82E16814130781 ) that is something like 4 or 5 times faster than my old Tesla, for half the price!

Yeah, most of my experiments have been done on gaming cards too.  I'm running a 480 and a 580 right now.  Strangely, when someone in our lab tried a 680 he said it was quite a bit slower than his 580.  I'm not sure what the explanation is, or if the finding would generalize beyond his particular code (highly optimized convolution net stuff in raw CUDA).

> I hear you on being skeptical about the asynchronous updates. I haven't delved into that but will let you know if it turns out to be useful. I was a bit embarrassed after sending you my modified code to realize that I had misinterpreted the "drop" variable to have something to do with dropout. I have since spent much more time with the code and understand it much better. The main issue I was having was that the trained model would start over-fitting pretty quickly on the validation data, even using an L2 prior on the weights. I implemented early stopping, but that seems to be a very crude method. So I investigated trying to do something like dropout with your code. Since your method is deterministic, I'm not sure if injecting stochasticity with dropout is even sensible. Still, I tried adding it in two ways.

Well, it's not completely deterministic.  It does like to use largish minibatches though.  There is a tradeoff with the size of the minibatch and the number of CG steps (see the recent document on my website).

> The first is like the method described in the Hinton paper, but instead of randomly selecting half of  all the hidden units, I only sample from the non-zero ones, since you are using sparse initialization. That method seems to be of questionable value based on my tests, although I'm still not 100% sure. The other method I tried is to randomly pick a percentage of the weights and to then randomly perturb them by either dilating or contracting by a small factor (e.g., multiplying by 0.99 or 1.01). This is not done to the actual weights to avoid excessive "corruption"; they are instead copied to a temporary variable right after the " rdEdy{ii+1} = [];" line of the "computeGV" function and then the random perturbations are applied to that variable. The perturbed weights are then used to compute " rdEdy{ii} ". I also did something similar to the inputs themselves as they are formed into mini-batches, so that different perturbations will be applied for every epoch.

The sparse initialization is only making the weights sparse.  And they quickly become non-sparse after a bit of optimization.  The sparseness is there only to avoid saturating the unit (while not being force to divide the weights by huge constant - although that can work okay if done properly, especially with TANH units - see Glorot & Bengio, 2010).

You could in principle just compute the gradient and GV products with a random set of units randomly dropped out for each case in the current minibatch (using as different set for each case).  And you would probably want to keep those particular dropped sets constant during each CG run.

In terms of preventing overfitting, I'm hopefully going to start exploring the connections between Bayesian methods and sampling.  I think this will also help with global optimization performance (I have some pretty good evidence for this already). But other than that kind of speculative stuff, there are other regularizers to try in addition to L2 and drop-out.  Swersky et al (2010?) looks at some regularization terms that arise naturally when looking at score matching.  These can be computed for deeper nets using the "Curvature Propagation" method, for example (see my website).  Also, you might want to consider expanding you dataset by augmenting it with randomly perturbed/transformed cases, and maybe try integrating unsupervised learning somehow.  For example, you could try training an autoencoder and then putting a several layer classification net on the resulting codes (assuming the codes themselves don't overfit - which you can ensure by making them low dimensional, or maybe by using contractive penalty terms [see Rifai et al's papers]).  But it's hard for me to make any more specific recommendations without knowing the nature of your task.

> On top of that, I pulled out several hard-coded values in your code and made them parameters of the function: the number of incoming connections per unit, the lower bound for the value of rho to apply the damping heuristic, the initial lambda, and the maxiters. I also made as parameters of the function the various quantities described earlier for my attempt at introducing dropout. Then I used CMA-ES...

Is this method similar to "Bayesian optimization"? Many people in our lab have been using it. Personally I still like to do things by hand, but I also agree with your point that it can find things you wouldn't have thought of, and that you can then later investigate and try to understand what it finds.

> Anyway, hopefully you find these explorations at least moderately interesting.

It sounds like you are trying some interesting things.  Let me know how things turn out.

Cheers,
James

---

**From:** Jeffrey Emanuel  
**To:** James Martens  
**Date:** December 21, 2012 2:42 AM

Thank you for your quick and detailed reply. I'll check out the papers you suggest and explore the various other regularization options. But I suspect that all of them (if they work at all) are fundamentally equivalent to some kind of dropout or random corruption of the model state. I believe that is a key feature of how real brains work. And yes, I am randomly perturbing my input data on the fly as it is turned into mini-batches (that way there's an unlimited supply of random corruption without having to create separate prepared data). 

As for the GPU, this thing is awesome. Seriously, read the reviews on this thing on newegg (they will make you laugh, it is nuts how hardcore these gamers are). Your friend in the lab who had poor performance must have something wrong with his CUDA setup, flaky older drivers (new ones were just released a couple weeks ago with the release on 12/12 of the new Tesla cards (if you have $3,500 to spend, that's even more amazing, and has much more RAM (which is what really matters-- I can't do things more than 13 layers or more than a few thousand units (less than 5,000 is fine) unless I am using the Tesla card, despite it being way slower (448 cores compared to like 3,000 cores on the 690). This is Moore's law on crack. 

I try to use mini-batches of at least 15,000 rows (my data has about 500 -1500 columns) with your code. To me, that is the single biggest benefit of the HF method. Pre-training works great if you know what parameters you want to use already. Because once you spend 20 hours to train a DBN using a super small batch size to get good results and a super low learning rate (I came up with a simple adaptive gradient function that helped with that), you are locked into that architecture except for being able to throw out all the layers after a given layer. So if I want to explore the space of hyper-parameters, each evaluation to that "functional" might take a whole night to do. But with your code, I can sometimes get good results in a few minutes. More importantly, I can bail out early if it's not looking promising, whereas I had to waste night after night trying to optimize the DBN architecture so that I could smoothly scale down the reconstruction error at each epoch (to "share the work" of the representation somewhat evenly across the layers). 

What I am doing with CMA-ES is sort of a ghetto version of Bayesian optimization. But that's just because I made my loss function based on the validation performance, and optimizing generalization power is basically the same thing. My actual loss function now is a bit more involved (not sure yet if I should keep it simple or try to do some multi-objective stuff):

        objective_score = -( 1000*(validation_performance_history(end) + 4*median_improvement)) / max([5 log(duration_of_run_in_seconds)] ) / max([7 log(sum(layersizes))]); 

The idea is to find networks that perform well, and which show a steady history of improving on each training epoch. But I also want to favor ones that don't get stuck in these horrible CG loops where you can waste 20x the normal epoch time (I am thinking of a way of breaking out of those automatically, since they almost always result in a crappy step anyway), so I penalize the execution time. Finally, I want to optimize memory and get the most per neuron, so I also penalize for number of neurons.

Anyway, CMA-ES is really incredible. I attached a PDF explaining it here. It basically gives you total freedom to try anything, and if it is silly or crazy and doesn't help at all, then CMA-ES just won't select for it over time. But if it helps it certain cases and hurts in others, then it will also learn that dependency in the input parameters. 

Let me give you another idea, which you will probably think is pretty crazy. I understand that the weight initialization starts out sparse and then becomes more evenly distributed during the course of the optimization, but I had the idea that maybe the "structure" of the initial pattern could make a difference. If it somehow provided a scaffolding that could result in structured functional areas instead of just randomly wired in any old way, then maybe this could help performance in some way, either speeding up convergence or allowing generally better performance/accuracy. I basically wanted to try all sorts of patterns-- but how would you know what would work well a priori? So I had the idea to use one dimensional cellular automata (the kind Wolfram wrote about in his book) to initialize all sorts of different patterns (very simple, simple repeated/nested patterns, and more complicated class 3 and 4 CAs that look organic, I also made it so I could control the amount of random noise, density, etc. Anyway, I just put it in as an alternative to sparse initialization as follows:

```matlab
 %Instead of random sparse weight initialization, use a cellular automata as the basis of the sparsity pattern:
            outputString( 'Using CA based weight initialization...' );
            ca_output_continuous_values = 0;
            ca_show_plots = 0;
            start_rows = 5000;
            packed_parameter_vector = zeros(psize,1); %not mzeros
            [Wtmp,btmp] = unpack(packed_parameter_vector);
            for ii = 1:numlayers
                tic
                initcoeff = 1;
                num_rows = layersizes(ii);
                num_cols = layersizes(ii+1);
                incremented_ca_rule_number = ca_rule_number + (ii-1)*ca_rule_number_incrementer;
                pattern = ca_weight_initialization(num_rows,num_cols,ca_desired_weight_density_rate,ca_random_factor,incremented_ca_rule_number,start_rows,ca_invert_output,ca_resize_scale,ca_output_continuous_values,ca_show_plots,fid);
                random_weights = randn(num_rows,num_cols)*initcoeff;
                Wtmp{ii} = random_weights .* pattern;
                CA_time = toc;
                outputString( ['Finished layer ' num2str(ii) ' of ' num2str(numlayers) ' (' num2str(num_rows) 'x' num2str(num_cols) ') in ' num2str(CA_time,3) ' seconds.'] );
                fprintf('\n');
            end
            packed_parameter_vector = pack(Wtmp, btmp);
            clear Wtmp btmp
```

I also attached the function and a script that calls it. I just finished it, so I don't know if it was a waste of a couple hours, but hopefully I will know in a few days. Because if it helps, then I will probably find a model that works way better than what I have now, and it will keep exploring the space of initialization patterns to find even better ones. There are so many options in terms of the different CA rules, the re-sizing factor, and the other options that it will take weeks probably to really explore them. But that's the kind of thing you can do with this method. The beauty of this stuff is that we have real time feedback on how well it is working by looking at the generalization power on validation data.

I also thought more about the asynchronous stuff and I think I can make it work, and yes I think it could be qualitatively different because you could search REALLY big nets that would take an hour for one epoch on a single machine. I agree with you that you can't linearly combine the weights in a simple naive manner. But I think if you keep track of the state of the system, and each processing node in the cluster had its own "latest copy" of the parameters (that is really slightly out of date), then each node could compare the current state of the canonical copy of the parameters (sitting in the dropbox .mat file) to the last state of the canonical copy that it remembered seeing. Then it would look for the entries that changed since it was busy doing its last epoch of training. Then, the node looks at the changes between the new parameters that it has just optimized for a single epoch and the state of the parameters from its previous epoch. Then, I think if the node avoids updating any entries that changed since the last time it deposited its parameters in the previous epoch, it will effectively be optimizing a different part of "weight space" and I think that should be fine (i.e., the system should be pretty robust to that sort of thing). The problem would be if the nodes diverge and then start fighting each other and trying to move the weights in opposing directions. It will make more sense once I code it, but I think it is pretty clear. I think I can take care of race/lock conditions (where multiple nodes are trying to hit the file at the same time) by simply generating little text files in the dropbox folder with the name of the machine that is currently updating the file. If there is no text file, then the node is free to update it (which it will do after generating its own text file to lock the file). Hopefully I will get that done tomorrow. 


-Jeff

---

**From:** James Martens  
**To:** Jeffrey Emanuel  
**Date:** January 5, 2013 7:47 PM

Hi Jeff,

Sorry for the late reply. I was away for the holidays and am only now catching up with my email backlog.

> I suspect that all of them (if they work at all) are fundamentally equivalent to some kind of dropout or random corruption...

Well, I tend not to agree that they are equivalent. Regularization is a complex problem which I can't imagine having one solution that subsumes the others. In some sense, it is equivalent to model selection, which is probably 50% of what ML is about.

And drop-out is actually pretty different from a system where noise is part of the model (like a "sigmoid belief net" for example).

> As for the GPU, this thing is awesome. Seriously, read the reviews on this thing on newegg (they will make you laugh, it is nuts how hardcore these gamers are). Your friend in the lab who had poor performance must have something wrong with his CUDA setup, flaky older drivers (new ones were just released a couple weeks ago with the release on 12/12 of the new Tesla cards...

Hmmm.. So to be sure, have you compared directly to a 580?  People in my lab would be interested to know if 680s were actually faster for the stuff they do.

The 690 is a 2 GPU in 1 card, isn't it?

> I try to use mini-batches of at least 15,000 rows (my data has about 500-1500 columns) with your code. To me, that is the single biggest benefit of the HF method. Pre-training works great if you know what parameters you want to use already. Because once you spend 20 hours to train a DBN using a super small batch size to get good results and a super low learning rate...

Recently people in our lab has been able to get good results training certain deep models with carefully tuned SGD with momentum (also carefully tuned, with the constant getting very large, etc). And you need to use a good initialization of course. How hard have you tried these kinds of more classical methods in your application?

> What I am doing with CMA-ES is sort of a ghetto version of Bayesian optimization. But that's just because I made my loss function based on the validation performance, and optimizing generalization power is basically the...

Sounds neat. Let me know how it goes.  It's hard to predict how well something like this would do since noone I think really understands what happens in non-convex optization of neural nets beyond vague and often wrong intuitions. Although, if I had to guess I would say that the initializations probably won't make a huge difference once they are already "good enough" (i.e. well scaled, no saturation, reasonable variety of activation patterns and lack of degenerate symmetries), which is based on my experience trying different initializations.

A while ago I did some experiments that showed how even an all-zero initialization can produce good results if you have the right kind of noisy optimizer/sampler system (see the "Bayesian neural nets" done by Radford Neal in the 90's).

> I also thought more about the asynchronous stuff and I think I can make it work, and yes I think it could be qualitatively different because you could search REALLY big nets that would take an hour for one epoch on a single...

Sounds exciting.  Just make sure you compare to the more naive solution of just distributing the gradient computation over different processors and combining them centrally to produce a low-noise gradient estimate.  Even just in the context of simple SGD. The problem I have with these asynchronous methods is that if there really is an advantage to performing sequential updates, I feel like this could be really messed up when you combine things. And then you might as well just do a distributed gradient computation.  Do you know if there is any evidence one way or the other in the literature?

---

**From:** Jeffrey Emanuel  
**To:** James Martens  
**Date:** January 23, 2013 11:22 PM

Hi James,

First, let me apologize for the extreme lateness of my response. I kept meaning to do it, but I hand't yet done the bench marking of the GPU and always had some reason why I didn't want to stop a calculation. Finally I decided to just respond without the benchmark. The thing is, I have the 690 card in a different machine with quite different specs from my machine with the 580, so a benchmark wouldn't be directly comparable anyway. I can tell you that I strongly believe it is significantly faster (at least 40-50%) on the sorts of computations that are used in the HF method. I am also overclocking it a bit, which seems to help. I also think it is very important to have fast RAM in the machine to really get the full benefit of the faster GPU.

See the rest of my response below.

> Well, I tend not to agree that they are equivalent.  Regularization is a complex problem which I can't imagine having one solution that subsumes the others.  In some sense, it is equivalent to model selection, which is probably 50% of what ML is about. And drop-out is actually pretty different from a system where noise is part of the model (like a "sigmoid belief net" for example).

You are probably right. I read a very nice paper about corruption models which you have probably seen (by Laurens Maaten, http://www.cse.wustl.edu/~kilian/papers/ICML2013_MCF.pdf ). There are clearly lots of possible ways to add corruption/noise to the state of a system (both that actual data and also to the parameters of the model), and they seem to do different things. I think it also doesn't have to be an either/or decision-- you can use dropout (I think I prefer the term blankout used in Maaten's paper) and also add noise the input data. I especially like the idea of "scale sensitive" corruption, which takes into account the magnitude of the input, so that you can think of it more as a dilation/contraction by a random percentage of the input. I have taken a pretty open-ended approach to the question, where I try several plausible methods and let them "duke it out" using CMAES to pick the hyper-parameters.

> Hmmm.. So to be sure, have you compared directly to a 580?  People in my lab would be interested to know if 680s were actually faster for the stuff they do. The 690 is a 2 GPU in 1 card, isn't it?

Yes, that's correct. Which is annoying if you are using Matlab, because it would take up 2 Jacket licenses. Even worse is that you can longer buy new licenses to Jacket since they were purchased by the Mathworks. Luckily I was grandfathered in because of my 2 existing licenses, and they let me add an extra GPU for free. It will be much better in the long-run to not have to shell out so much for Jacket though-- I'm sure it has taken pound of flesh from the ML community in the last few years. 

> Recently people in our lab has been able to get good results training certain deep models with carefully tuned SGD with momentum (also carefully tuned, with the constant getting very large, etc). And you need to use a good initialization of course. How hard have you tried these kinds of more classical methods in your application?

Yes, I read Ilya's thesis recently and he talks about how to construct good momenta schedules. I have tried those methods (and made up a couple of my own), and I still don't think it compares to the results I am getting using HF. I have also been surprised at the different levels of performance that result from changing the HF parameters. For example, the replacement rule for the dampening constant can be much larger or smaller, and still get good (and often different) results. 
> Sounds neat.   Let me know how it goes.  It's hard to predict how well something like this would do since noone I think really understands what happens in non-convex optization of neural nets beyond vague and often wrong intuitions.  Although, if I had to guess I would say that...

Thanks. I think the jury is out for me still on how important the initialization is. I think it depends on what you want from a learning algorithm. If you want something that always gives consistently good results and rarely fails (comes up with very poor results or doesn't converge), then you will pick a random initialization because that way you can put a lower bound on how bad it can be. But I think that also limits how good it can be, and in fact you may be better served by using a more brittle procedure that is very sensitive to small changes in the hyper-parameters, and simply running it many thousands of times and grading/collating the results into a single output. There are a lot of problems in the world where simply getting an answer would be great, even if it requires a lot of time and effort to search a big space looking for settings that really work well for your problem. There are also problems where you want it to just work, like speech recognition on a phone. I am generally more excited and interested in the really hard problems where getting any kind of sensible result would be nice. 
 
> A while ago I did some experiments that showed how even an all-zero initialization can produce good results if you have the right kind of noisy optimizer/sampler system (see the "Bayesian neural nets" done by Radford Neal in the 90's).

I believe it. You just have to get the temperature up from absolute zero so the particles can start flying around (I often lapse into the physical interpretation when I think about these kinds of models) and then the learning algo can start steering them in the right direction.  

> Sounds exciting.  Just make sure you compare to the more naive solution of just distributing the gradient computation over different processors and combining them centrally to produce a low-noise gradient estimate.  Even just in the context of simple SGD.  The problem I have with these asynchronous methods is that if there really...

Yeah, I always try to compare any new code to the previous best results. My approach is to take the time I save from speculating about what will work better (I agree that this stuff is not intuitive at all, and you are likely to be wrong more than right) and instead just hack out some code and do a quick experiment. For now I have de-emphasized working on the asynchronous stuff, mostly because the part that tripped me up was boring race-condition stuff, which is just engineering stuff. I also realized that the advantages of the asynchronous model, given the way the hardware is set up and the relative speeds of RAM/SSD/ethernet, is likely not very pronounced until you get up to a Google scale (BTW, you should see the latest presentation from Jeff Dean, they are really breaking new ground there: http://i.stanford.edu/infoseminar/dean.pdf ; I can't help but think that Skynet is near). 

I'm not familiar enough with the literature to really give you a good answer. As I said, my interest lately has moved on to other areas. I have been spending a lot of time on an interesting problem of integrating thousands of pre-computed models that are separately graded based on validation performance. There are lots of simple ad-hoc ways to form an ensemble from a bunch of models, and I explored many different ones. But ultimately I realized that the whole thing can be viewed as another, secondary machine-learning problem. That way you can learn relationships among the various sub-models, which could give you more information than what you started with. For example, you can learn that if you had 3 models, and the predicted labels for the model with the best validation performance disagree with the predictions from the second and third best performing models, but the second and third model predictions agree with each other, that you are better off deferring to the second and third best models instead of blindly choosing the single best model. So if you think of the models as being a team of people, the secondary learning model is like a manager who knows how to get the best out of each model, how to play them off each other to get to the truth, etc. Ultimately I don't really like having any parameters that a human has to set, even if that means that it takes months of computations before you get something that works. I view it more as a search problem than anything else-- exploring the world of possible models in an efficient way and leveraging the knowledge that you get from that exploration in addition to just using the underlying models. 

---

**From:** James Martens  
**To:** Jeffrey Emanuel  
**Date:** February 7, 2013 5:17 PM

> First, let me apologize for the extreme lateness of my response. I kept meaning to do it, but I hand't yet done the bench marking of the GPU and always had some reason why I didn't want to stop a calculation. Finally I decided to just respond without the benchmark. The thing is, I have the 690...

No worries.  I've been quite busy myself. Mostly obsessing over this this annoyingly hard open problem in lower bound while letting my other obligations fall by the wayside.

Well, the colleague of mine who tried it out had a pretty beefy machine and he actually found that performance didn't simply stay the same, but actually got significantly worse. Perhaps though there was a flaw in his tests.  I'll ask him if he has since tried again. 

The 690 is a dual GPU card right? Perhaps your code is making use of both GPUs, but would actually be slower than the 580 if you ran it only on one of the two GPUs.

> You are probably right. I read a very nice paper about corruption models which you have probably seen (by Laurens Maaten, http://www.cse.wustl.edu/~kilian/papers/ICML2013_MCF.pdf ). There are clearly lots of possible ways to add corruption/noise to the state of a system (both that actual data and also to the parameters of the model), and...

That is probably the best strategy if what you care about is performance.  I'm a bit of a purest when it comes to these things though :)

> Yes, that's correct. Which is annoying if you are using Matlab, because it would take up 2 Jacket licenses. Even worse is that you can longer buy new licenses to Jacket since they were purchased by the Mathworks. Luckily I was grandfathered in because of my 2 existing licenses, and they let me add an extra GPU for free. It will be much better in the long-run to not have to shell out so much for Jacket though-- I'm sure it has taken pound of flesh from the ML community in the last few years.

Yeah I bought one of their expensive 2 GPU licenses last year after having my old ones expire.  Thankfully this license won't. But ironically I haven't actually been using it much since then since my interests have shifted somewhat to theoretical topics (although I'll probably come back soon enough).

> Yes, I read Ilya's thesis recently and he talks about how to construct good momenta schedules. I have tried those methods (and made up a couple of my own), and I still don't think it compares to the results I am getting using...

Yeah I too found that long-term performance can be highly sensitive to these kinds of things.  Which is unfortunate since it seems hard/impossible to predict ahead of time.

> I believe it. You just have to get the temperature up from absolute zero so the particles can start flying around (I often lapse into the physical interpretation when I think about these kinds of models) and then the learning algo can start steering them in the right direction.

Although it's not obvious that this will always work. I was personally surprised by these results since I thought that the zero-init would simply be too far away from good solutions (or separated by nearly impossible to pass energy barriers).

> Yeah, I always try to compare any new code to the previous best results. My approach is to take the time I save from speculating about what will work better (I agree that this stuff is not intuitive at all, and you are likely to be wrong more than right) and instead just hack out some code and do a quick experiment. For now I have de-emphasized working on the asynchronous...

I occasionally talk to the guy who is doing most of the technical work on that project (Marc'Aurelio Ranzato). He was a post-doc at our lab a couple years ago. I think he is in the process of determining which aspects of their previous work was important and which aspects were superfluous. I probably should ask him about how important this asynchronous stuff is beyond simply allowing more data to be processed.

In terms of Skynet though, I think we have a long way to go still :) Even if you get a good system for perception, you still have to solve the problem of giving these systems some kind agency or "will" to do things.

> I'm not familiar enough with the literature to really give you a good answer. As I said, my interest lately has moved on to other areas. I have been spending a lot of time on an interesting problem of integrating thousands of pre-computed models that are separately graded based on validation performance. There are lots of simple ad-hoc ways to form an...

Yeah these problems are tricky.  And unlike with perception, I doubt we can get a lot of inspiration from the brain since brains don't really seem to be solving this kind of problem. Perhaps something of a more combinatorial flavor like a decision tree would work best? Who knows.

Regards,
James

---

**From:** Jeffrey Emanuel  
**To:** James Martens  
**Date:** May 6, 2013 12:53 PM

Hi James,

Hope you're well. I was investigating some performance tuning in the Matlab HF code for when it is running on the CPU (not all of my machines have GPUs/Jacket licenses). I did some profiling on the code and found the bottleneck lines. I have been playing around with Matlab Coder product recently, using the latest Matlab (2013a) and a new compiler (MS VS 2012). Matlab Coder is a pretty amazing product. I can't program in C++ for my life, and yet I was able to automatically generate highly optimized code pretty quickly that I compiled into 64bit windows mex files.

Basically I made two mex files that can be easily "dropped in" to your existing code as shown below. I found there to be a very large speedup, although I haven't measured it. I can send you the C files and everything you need to compile mex files for linux if you would like, but attached are the Windows mex files. Hopefully these might be of some use to you. I am excited to see if Matlab will support the Intel Xeon Phi in release 2013b. If they do, I think it could permanently tilt the balance away from using the nvidia cards for this sort of thing. Anyway, let me know if you have any questions.

1) computeLL_helper_func_mex
   
```matlab
yi = conv(in(:, ((chunk-1)*schunk+1):(chunk*schunk) ));

outc = conv(out(:, ((chunk-1)*schunk+1):(chunk*schunk) ));

for ii = 1:numlayers
%xi = W{ii}*yi + repmat(b{ii}, [1 schunk]);
if use_jacket
  xi = bsxfun(@plus,W{ii}*yi,b{ii});
  if strcmp(layertypes{ii}, 'logistic')
    yi = 1./(1 + exp(-xi));
  elseif strcmp(layertypes{ii}, 'tanh')
    yi = tanh(xi);
  elseif strcmp(layertypes{ii}, 'linear')
    yi = xi;
  elseif strcmp(layertypes{ii}, 'softmax' )
    tmp = exp(xi);
    %yi = tmp./repmat( sum(tmp), [layersizes(ii+1) 1] );
    yi = bsxfun(@rdivide,tmp,sum(tmp));
    tmp = [];
  end
else
  [xi,yi] = computeLL_helper_func_mex(W{ii},yi,b{ii},layertypes{ii},layersizes(ii+1),schunk);
end
```

2) gradient_comp_helper_func_mex
```matlab
yi = conv(y{chunk, ii});

if hybridmode && chunk ~= targetchunk
  y{chunk, ii} = []; %save memory
end

if use_jacket
  %standard gradient comp:
  dEdW{ii} = dEdxi*yi';
  dEdb{ii} = sum(dEdxi,2);

  %gradient squared comp:
  dEdW2{ii} = (dEdxi.^2)*(yi.^2)';
  dEdb2{ii} = sum(dEdxi.^2,2);
else
  [dEdW{ii} ,dEdb{ii},dEdW2{ii},dEdb2{ii} ] = gradient_comp_helper_func_mex(dEdxi,yi);
end
```
---

**From:** James Martens  
**To:** Jeffrey Emanuel  
**Date:** May 6, 2013 6:17 PM

Cool, thanks.
How much of a speed-up do you get from using these?

James

---

**From:** Jeffrey Emanuel  
**To:** James Martens  
**Date:** May 7, 2013 4:04 PM

It actually depends a lot on the size of the network. I probably should have explained that I specified a pretty small maximum size for the dimensions of the data, so it could be very smart about the memory management. I have been working a lot with very small networks, I can I can just see that there is a big step up for those. But since most of the interesting models are much larger, there is probably much less of a benefit (it is still clearly better on the GPU). I haven't really done much benchmarking because I am crazy busy now and also because I have changed so much stuff that it is not really the same code base anymore. I will try to include benchmarks next time I send you something like this though!

Take care.

