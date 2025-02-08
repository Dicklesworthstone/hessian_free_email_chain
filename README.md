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

Thanks, and good luck. I'll be interested to know how it turns out.

Cheers,
James

---

**From:** Jeffrey Emanuel  
**To:** James Martens  
**Date:** December 21, 2012 2:42 AM

Thank you for your quick and detailed reply. I'll check out the papers you suggest and explore the various other regularization options. But I suspect that all of them (if they work at all) are fundamentally equivalent to some kind of dropout or random corruption of the model state. I believe that is a key feature of how real brains work. And yes, I am randomly perturbing my input data on the fly as it is turned into mini-batches (that way there's an unlimited supply of random corruption without having to create separate prepared data).

---

**From:** James Martens  
**To:** Jeffrey Emanuel  
**Date:** January 5, 2013 7:47 PM

Hi Jeff,

Sorry for the late reply. I was away for the holidays and am only now catching up with my email backlog.

> I suspect that all of them (if they work at all) are fundamentally equivalent to some kind of dropout or random corruption...

Well, I tend not to agree that they are equivalent. Regularization is a complex problem which I can't imagine having one solution that subsumes the others. In some sense, it is equivalent to model selection, which is probably 50% of what ML is about.

And drop-out is actually pretty different from a system where noise is part of the model (like a "sigmoid belief net" for example).

---

**From:** Jeffrey Emanuel  
**To:** James Martens  
**Date:** January 23, 2013 11:22 PM

You are probably right. I read a very nice paper about corruption models which you have probably seen (by Laurens Maaten, http://www.cse.wustl.edu/~kilian/papers/ICML2013_MCF.pdf). There are clearly lots of possible ways to add corruption/noise to the state of a system (both that actual data and also to the parameters of the model), and they seem to do different things. I think it also doesn't have to be an either/or decision-- you can use dropout (I think I prefer the term blankout used in Maaten's paper) and also add noise the input data. I especially like the idea of "scale sensitive" corruption, which takes into account the magnitude of the input, so that you can think of it more as a dilation/contraction by a random percentage of the input. I have taken a pretty open-ended approach to the question, where I try several plausible methods and let them "duke it out" using CMAES to pick the hyper-parameters.

---

**From:** James Martens  
**To:** Jeffrey Emanuel  
**Date:** February 7, 2013 5:17 PM

On Wed, Jan 23, 2013 at 11:22 PM, Jeffrey Emanuel wrote:

That is probably the best strategy if what you care about is performance. I'm a bit of a purest when it comes to these things though :)

The 690 is a dual GPU card right? Perhaps your code is making use of both GPUs, but would actually be slower than the 580 if you ran it only on one of the two GPUs.

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

---

**From:** James Martens  
**To:** Jeffrey Emanuel  
**Date:** January 5, 2013 7:47 PM [Earlier thread]

> Yes, that's correct. Which is annoying if you are using Matlab, because it would take up 2 Jacket licenses. Even worse is that you can longer buy new licenses to Jacket since they were purchased by the Mathworks. Luckily I was grandfathered in because of my 2 existing licenses, and they let me add an extra GPU for free. It will be much better in the long-run to not have to shell out so much for Jacket though-- I'm sure it has taken pound of flesh from the ML community in the last few years.

Yeah I bought one of their expensive 2 GPU licenses last year after having my old ones expire. Thankfully this license won't. But ironically I haven't actually been using it much since then since my interests have shifted somewhat to theoretical topics (although I'll probably come back soon enough).

> I'm not familiar enough with the literature to really give you a good answer. As I said, my interest lately has moved on to other areas. I have been spending a lot of time on an interesting problem of integrating thousands of pre-computed models that are separately graded based on validation performance...

I occasionally talk to the guy who is doing most of the technical work on that project (Marc'Aurelio Ranzato). He was a post-doc at our lab a couple years ago. I think he is in the process of determining which aspects of their previous work was important and which aspects were superfluous. I probably should ask him about how important this asynchronous stuff is beyond simply allowing more data to be processed.

In terms of Skynet though, I think we have a long way to go still :) Even if you get a good system for perception, you still have to solve the problem of giving these systems some kind agency or "will" to do things.

These problems are tricky. And unlike with perception, I doubt we can get a lot of inspiration from the brain since brains don't really seem to be solving this kind of problem. Perhaps something of a more combinatorial flavor like a decision tree would work best? Who knows.

Regards,
James
