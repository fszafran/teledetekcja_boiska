"""
Zalozenia:
 - Boisko w NIR maja wartosci w zakresie XXXX - YYYY
 - Boisko ma pp = 1860 m^2
 - Boisko ma wymiary 62m x 30m
 - Boisko jest prostokątem
 - Pixel ma wymiar 3.3x3.3
* wszystkie wartosci traktujemy z ustalona tolerancja bledow

Algorytm:
1. For loop po calym obrazie
2. Jezeli pixel miesci sie w zakresie XXXX - YYYY -> BFS z sumowaniem powierzchni, jezeli powierzchnia koncowa za duza -> skip, 
za mała nie zapisuj do wyniku. Zapisywanie przetworzonych pikseli jako odwiedzonych, żeby nie wchodzic znowu w bfs.
!(pod uwage bierzemy tylko pixele spełniajace wymagania zakresu wartosci)!
3.Koniec loopa -> wynik to lista list pixeli składajacych się na kandydatów 
4.Loop po listach z kandydatami
5.Dla każdej listy -> graham scan z ktorego mamy boundry niezalezne od orientacji obiektu
6.Poszukiwanie 2 par wzajemnie równoległych prostych w convex hullu, jezeli nie ma -> skip, 
jeżeli są porównaj ich długosci z wymiarami boiska, zapisz jako kandydata.
** cv2.convexHull + cv2.minAreaRectangle + porównanie pól convexhull i minAreaRectangle?
7. wedlug otrzymanego obrysu (convex hull) stworz warstwe geojson z poligonami.
"""
