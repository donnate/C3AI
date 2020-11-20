# C3AI Challenge: Predicting Heterogeneous Individual Probabilities of Infection based on Personalized Information

__Short Abstract__ When will we be able to safely travel again or gather together in public? In short, when we can determine that the risks are acceptably low. Unfortunately, most models that predict the risk of infection at public gatherings fail to truly capture the heterogeneity in the infectiousness and susceptibilities of the people present, yet the spread of infectious diseases are dominated by tail events. Our Bayesian Stochastic Expectation-Maximisation method provides personalized estimates for each individual’s infectious and susceptibility statuses by combining data on location, behavioural factors, symptoms, and various COVID-19 test results, to inform individual decisions and public policy. 

Check out our shinyapp at [the following link!](https://homecovidtests.shinyapps.io/C3AI_AntigenTesting/)

Disclaimer: this is not a medical diagnostic tool. This is simply to provide a little it of guidance, and to see -- for instance -- if the test that you have taken is likely to be a false negative, or to give you an idea of the risk in your neighborhood.

## I. Motivation

COVID-19 has claimed almost 1.5 million lives, yet it’s reach is much greater: affecting the lives and livelihoods of billions as we attempt to curtail its spread through lockdowns and restrictions. This relatively indiscriminate approach has led to mass unemployment and social isolation, both of which may have untold consequences on our health and wellbeing. But COVID-19 does discriminate: demographics, individual health characteristics, behavioral and geographical data – all considerably alter one’s risk of catching and spreading the virus. Our response to the crisis should thus recognize and control for this heterogeneity by leveraging the wealth of COVID-19 studies to create reactive, granular and even personalised recommendations which help individuals (and by extension society), return to normal life – a challenge that we take on in this submission through the scope of AI-informed COVID testing and monitoring at scale.

In short, the problem that we are facing is two-fold:
(a) COVID has shown that there is a substantial amount of variability in people's propensity to catch (and spread) the disease. These could be due to different lifestyle or risk-exposure factors: healthcare workers are much more exposed to potential contagion than someone working from home. We thus need to account for this variability in our prediction model.
(b) On the other, despite the wealth of available information, it is hard to identify who is more likely to be infectious:
   - Tests are unreliable. In particular, even RT-PCR tests (which are taken to be the gold standard) are far from having 100% sensitivity. In fact, as highlighted in [this recent paper](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7240870/), their sensitivity is highly contingent on time since symptom onset.
   - Datasets that could allow to identify risk factors are but only weakly informative:
        + there is often a sample bias in the population that these studies are considering.
        + it is hard to find consistent variables to pool studies (which could help alleviate sampling bias).


So why the Bayesian formalism? The method that we propose here is indeed based on a Bayesian generative model. The idea is the following:
- This generative probabilistic framework helps us pool mutiple datasources
- It helps us deal with missing data
- It is also essential in dealing with the bias that all the datasets are subject to.

In a way, we could think of this Bayesian network as a lego model. We train it, and replace the parts that are subject to sample bias to generalize to the general population by replacing it with another estimate (which we believe to be more accurate and generalizable --- in particular, in this study, we try to inform our network by published Bayesian meta analyses that leverage themselves multiple studies over the past few months).
