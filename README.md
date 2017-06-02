# US-Primary-Election-Predictions

The purpose of this code is to be able to succesfully predict both the candidate and the party that each county in the united states voted for in the primary elections based on their population statistics. 

I used a random forest algorithm to predict both the candidate and the party assoicated with each county. I used a cross validation test and determined that these predictions generated scores well into the mid 90's, so the actual erorr on the test set should be fairly low. I then used an approach to further reduce the error by sorting the candidates with misalligned parties and changing the parties if they met a certain threshold. 

There's plenty more analysis I could do on this data set in the future. One possilbity is taking my predictions and seeing if I can accurately predict the winner of the general election based on the parties that won each county.
