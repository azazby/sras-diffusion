# Selective Regional Adaptive Sampling for Diffusion Models

Exploring region selection strategies for improving Region-Adaptive Sampling (RAS) for diffusion transformers.

This project investigates how different region selection methods influence RAS, which is a novel diffusion sampling strategy that improves sampling efficiency by allocating more updates to fast-update regions, which are identified in the original method using predicted-noise standard deviation as a scoring heuristic. We explore alternative region-scoring functions that may better capture semantic importance or local detail, and evaluate how these choices influence update behavior and generated sample quality.
