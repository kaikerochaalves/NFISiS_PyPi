
Project description

Powered by Kaike Sa Teles Rocha Alves

NFISiS (new fuzzy inference systems) is a package that contains new machine learning models developed by Kaike Alves during his PhD research. 

    Website: kaikealves.weebly.com
    Documentation: Fourthcoming
    Email: kaikerochaalves@outlook.com
    Source code: https://github.com/kaikerochaalves/NFISiS_PyPi

It provides:

    the following machine learning models in the context of fuzzy systems: NMC, NMR, NTSK, GEN_NMR, GEN_NTSK, R_NMR, R_NTSK



Code of Conduct

NFISiS is a library developed by Kaike Alves. Please read the Code of Conduct for guidance.

Call for Contributions

The project welcomes your expertise and enthusiasm!

Small improvements or fixes are always appreciated. If you are considering larger contributions to the source code, please contact by email first.

To install the library use the command: 

    pip install nfisis

To import the NewMandaniClassifier (NMC), simply type the command:

    from nfisis.fuzzy import NewMamdaniClassifier

To import the NewMamdaniRegressor (NMR), simply type:

    from nfisis.fuzzy import NewMamdaniRegressor

To import the NTSK (New Takagi-Sugeno-Kang), type:

    from nfisis.fuzzy import NTSK

NewMandaniClassifier, NewMamdaniRegressor, and NTSK are new data-driven fuzzy models that automatically create fuzzy rules and fuzzy sets. You can learn more about this models in papers: https://doi.org/10.1016/j.engappai.2024.108155 and https://doi.org/10.1007/s10614-024-10670-w 

The library nfisis also includes the NTSK and NMR (NewMandaniRegressor) with genetic-algorithm as attribute selection. At this time, the paper containing the proposal of these models are fourthcoming.

To import GEN_NMR type:

    from nfisis.genetic import GEN_NMR

To import GEN_NTSK type:

    from nfisis.genetic import GEN_NTSK

Finally, one last inovation of this library that was part of the reasearch of the PhD of Kaike Alves and it is in his forthcoming thesis is the ensemble model with fuzzy systems, reffered as to R_NMR and R_NTSK:

    from nfisis.ensemble import R_NMR

    from nfisis.ensemble import R_NTSK

Once you imported the libraries, you can use functions fit and predict. For example:

    from nfisis.fuzzy import NTSK
    model = NTSK()
    model.fit(X_train, y_train)
    y_pred = model.predict(y_test)

If you want to look closely to the generated rules, you can see the rules typing:

    model.show_rules()

Otherwise, you can see the histogram of the rules by typing:

    model.plot_hist

The fuzzy models are quite fast, but the genetic and ensembles are still a bit slow. If you think you can contribute to this project regarding the code, speed, etc., please, feel free to contact me and to do so.
