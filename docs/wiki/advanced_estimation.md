## Advanced techniques for estimating degrees of freedom in a non central chi distribution

This section is mostly personal recommendations based on some literature, stuff I have played with
and stuff I have seen in MR physics classes. It is probably not exhaustive nor perfectly accurate,
but should give the interested reader a feeling of what is happening and why.

### The problem

Noise estimation in MR highly depends on
the reconstruction algorithm implemented by your vendor (*Dietrich et al.*). Unfortunately,
due to interference between adjacent receiver coils as used in modern parallel imaging
(i.e. pretty much always unless you are doing fancy specialized acquisitions),
the real noise distribution is slightly different than a pure Rician or
Noncentral chi distribution (*Aja-Fernandez et al., Constantinides et al.*).

It is still possible to estimate the distribution,
but the values of the 'standard deviation' and degrees of freedom of that distribution
depends on the parameters of the acquisition (i.e. SENSE maps, GRAPPA weights),
which are probably hard to acquire if you do not have a friendly MR physicist at hand
(they are always nice guys anyway, so don't be afraid to ask for help).
The authors of (*Aja-Fernandez et al.*) still offer a way to do a blind estimation of these values for
those interested to dig a bit more.

### Why it is hard to find a surefire correction method

Also based on my MR physics class understanding, due to the way the (closed source)
algorithm in each vendor's scanner software work, they are likely to discard
the signal coming from far away receiver elements from the imaged body region.
As an example, near the top of the head, the coil elements placed near the neck are very likely to measure
little relevant signal and mostly contribute noise.

This means that during the k-space reconstruction, the signal
contribution from these coils will get thrown away, and thus the number of effectively used coils will vary per region and will be lower than the number of coils on you receiver array. Add noise correlation into that
due to electrical interference and proximity of your receiver elements, and you are looking at some (hard to figure out) distribution which is different from what you expect according to the theory.

### What other people suggest

The authors of (*Veraart et al.*) also provided another way to estimate those
relevant parameters based on constructing synthetic noise maps for those
interested in trying another method.

As a final tl;dr advice, some other studies have found that for GRAPPA reconstruction,
with a 12 channels head coils N=4 (*Brion et al.*) \(that's what we use in Sherbrooke also
for the 1.5T Siemens scanner with a 12 channels head coil\)
works well and for a 32 channels head coils (*Varadarajan et al.*), a value around N=9 seems to work
(remember that N varies spatially, but it seems to be fairly homogeneous/vary slowly).

The authors of (*Becker et al.*) also indicates that in
the worst case, using N=1 for Rician noise is better than doing nothing.

Same observation from (*Sakaie et al.*) in real data; they fitted the background distribution and found out that for a sum of square (SoS) reconstruction with 12 coils, N = 3.76 ± 0.07 in 5 subjects (well, it should be an integer since it represents the number of degrees of freedom, so let's say N=4). As expected, it is much lower than 12 because of the correlation in each adjacent coils and produces DTI metrics (FA, MD, RD, AD) with a stronger bias than an adaptive combine (N = 1.03 ± 0.01) reconstruction.

### Take home message

In all cases, the take home message would be that estimating the real value for N
is still challenging, but it will most likely be lower than the number of coils present on
your receiver coil due to the way MRI scanners reconstruct and combine images.

From my personal recommendations (well, don't quote me too much on it though), a good rule of thumb would be for 12 coils -> N=4, for 32 coils -> N=8 and for 64 coils (which are separated as a 24 coils for the upper part and 40 coils for the lower part), I would try out N=6 as dictated by the upper part, N=4 if it does not produce satisfactory results.

## References

Aja-Fernandez, S., Vegas-Sanchez-Ferrero, G., Tristan-Vega, A., 2014.
Noise estimation in parallel MRI: GRAPPA and SENSE.
Magnetic resonance imaging

Becker, S. M. A., Tabelow, K., Mohammadi, S., Weiskopf, N., & Polzehl, J. (2014).
Adaptive smoothing of multi-shell diffusion weighted magnetic resonance data by msPOAS.
NeuroImage

Brion, V., Poupon, C., Riff, O., Aja-Fernández, S., Tristán-Vega, A., Mangin, J.-F., Le Bihan D, Poupon, F. (2013).
Noise correction for HARDI and HYDI data obtained with multi-channel coils and sum
of squares reconstruction: an anisotropic extension of the LMMSE.
Magnetic Resonance Imaging

Constantinides, C. D., Atalar, E., & McVeigh, E. R. (1997). Signal-to-Noise Measurements in Magnitude Images from NMR Phased Arrays. Magnetic Resonance in Medicine, 38(5), 852–857.

Dietrich, O., Raya, J. G., Reeder, S. B., Ingrisch, M., Reiser, M. F., & Schoenberg, S. O. (2008).
Influence of multichannel combination, parallel imaging and other reconstruction
techniques on MRI noise characteristics. Magnetic Resonance Imaging

Sakaie, M. & Lowe, M., Retrospective correction of bias in diffusion tensor imaging arising from coil combination mode, Magnetic Resonance Imaging, Volume 37, April 2017

Varadarajan, D., & Haldar, J. (2015).
A Majorize-Minimize Framework for Rician and Non-Central Chi MR Images.
IEEE Transactions on Medical Imaging

Veraart, J., Rajan, J., Peeters, R. R., Leemans, A., Sunaert, S., & Sijbers, J. (2013).
Comprehensive framework for accurate diffusion MRI parameter estimation.
Magnetic Resonance in Medicine