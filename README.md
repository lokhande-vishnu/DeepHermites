# Substituting ReLUs with Hermite Polynomials gives faster convergence for SSL

## Abstract
Rectified Linear Units (ReLUs) are among the most widely used activation function in a broad variety of tasks in vision. Recent theoretical results suggest that despite their excellent practical performance, in various cases, a substitution with basis expansions (e.g., polynomials) can yield significant benefits from both the optimization and generalization perspective. Unfortunately, the existing results remain limited to networks with a couple of layers, and the practical viability of these results is not yet known. Motivated by some of these results, we explore the use of Hermite polynomial expansions as a substitute for ReLUs in deep networks. While our experiments with supervised learning do not provide a clear verdict, we find that this strategy offers considerable benefits in semi-supervised learning (SSL) settings. We carefully develop this idea and show how the use of Hermite polynomials based activations can yield improvements in pseudo-label accuracies and sizable financial savings (due to concurrent runtime benefits). Further,we show via theoretical analysis, that the networks (with Hermite activations) offer robustness to noise and other attractive mathematical properties.

## Paper
The link to the arxiv version is here - https://arxiv.org/abs/1909.05479.pdf

## Code
The code for the experiments in the paper are available in the 'Code' directory

## Other Project particulars
The set of slides and the poster for the project are available in the main directory with the titles 'slides_MMLS19.pdf' and 'poster_MMLS19.pdf' respectively.
