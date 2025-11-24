# Selective Regional Adaptive Sampling for Diffusion Models

Exploring region selection strategies for improving Region-Adaptive Sampling (RAS) in diffusion models.

This project investigates how different region selection methods influence RAS for diffusion models. RAS improves diffusion sampling efficiency by allocating more updates to fast-update regions, identified in the original method using predicted-noise standard deviation as a scoring heuristic. We explore alternative region-scoring functions that may better capture semantic importance or local detail, and evaluate how these choices influence update behavior and generated sample quality.
