# Fazit

## Regression

Bei den Regressionsmodellen schneidet das Splines Modell von Patsy am besten ab. Die Ergebnisse nach Evaluierung durch die Testdaten sind um einiges schlechter als die Ergebnisse, welche durch die Trainings- oder Validationdatasets entstehen. Das liegt vermutlich daran, dass die Trainingsdaten zu sehr optimiert wurden, also zu viele Outlier entfernt worden sind.

Keines der Modelle erreicht den zu Beginn definierten Zielwert von einem niedrigen 5-stilligen RMSE.

## Klassifikation

Das Classification Modell von statsmodels erzielt in diesem Fall leicht bessere Ergebnisse als das von scikit learn und würde daher für das Deployment in eine produktive Umgebung ausgewählt werden.

Es fällt auf, dass die Abweichungen der Modellergebnisse zwischen Trainings- und Testdaten nicht so groß sind wie bei den Regressionsmodellen. Das liegt möglicherweise daran, dass hier nicht so viele Outlier entfernt wurden wie bspw. durch Cook's Distance oder Regression Diagnostics.

Auch das Ergebnis generell ist um einiges zufriedenstellender als das der Regressionsmodelle, was wohl aber an der Art der Vorhersage liegt (Es ist natürlich einfacher einen binären Wert wie Ja/Nein oder in diesem Fall über/unter vorherzusagen als einen exakten Preis).
