# Klasyfikacja znaków drogowych

Skrypt w języku Python pozwala na trenowanie i ocenę różnych modeli sieci neuronowych konwolucyjnych (CNN) do klasyfikacji znaków drogowych przy użyciu biblioteki TensorFlow i Keras. Udostępnia różne modele o różnych architekturach oraz opcje augmentacji danych.

## Wymagania

- Python 3.x
- TensorFlow
- Keras
- Matplotlib
- Numpy
- [Dataset](https://www.kaggle.com/datasets/kasia12345/polish-traffic-signs-dataset/data)

## Modele

Skrypt udostępnia następujące modele CNN:

1. **model01**: Podstawowy model klasyfikacji znaków drogowych bez warstw Dropout i mniejszą liczbą filtrów.
2. **model02**: Model klasyfikacji znaków drogowych bez warstw Dropout i większą liczbą filtrów.
3. **model03**: Model klasyfikacji znaków drogowych bez warstw Dropout, większą liczbą filtrów oraz więcej warstw konwolucyjnych.
4. **model04**: Model klasyfikacji znaków drogowych z warstwami Dropout, większą liczbą filtrów oraz więcej warstw konwolucyjnych.

## Augmentacja danych

Obsługiwane są dwa rodzaje augmentacji danych:

- **Podstawowa**: Augmentuje dane tylko przez zmianę skali obrazów.
- **Rozszerzona**: Augmentuje dane poprzez zmianę skali, skos, przybliżanie, rozjaśnianie, przesuwanie i zmianę kanałów obrazów.

## Wyniki

Po treningu skrypt generuje następujące wyniki:

- Pliki zapisanych modeli (jeśli użyto flagi `--save`).
- Histogramy dystrybucji danych treningowych i walidacyjnych.
- Wykresy dokładności treningu i walidacji.
- Wykresy straty treningu i walidacji.


# Flask GUI
[Flask GUI presentation](https://www.youtube.com/watch?v=R_KeMPKFlBg)
