# Application of optical flow for monitoring barge plume

- For Environmental Monitoring and Management Plan (EMMP), we are only interested in monitoring the *boundaries of of the plume*, and tracking how the plume as a whole moves and its *general* dynamics. We are not interested in the miniscule changes in velocity for each small region, only the *plume boundary*

- If we can create a mask to mask non-plume areas, then we can easily create a binary image, where we know that the optical flow contraint can be achieved.

- In that way, we don't have to worry about sunglint effects or radiometric correction that may introduce noise and significantly affect the brightness constancy constraint

- After which, we can also train an end-to-end U-net model to automate the calculation of the optical flow of the binary sediment plume
