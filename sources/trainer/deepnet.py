from ..dataset.dataset import Dataset


class Deepnet:
    def fit(self,
            train_set: Dataset, epoch: int,
            validation_set: Dataset | None = None
            ) -> list[float] | tuple[list[float], list[float]]:
        train_scores = [0.0 for _ in range(epoch)]
        val_scores = [0.0 for _ in range(epoch)]
        for i in range(epoch):
            for train_data, train_label in train_set:
                self._fit(train_data, train_label)
            train_scores.insert(i, self.score(train_set))
            if validation_set is not None:
                val_scores.insert(i, self.score(validation_set))
        if validation_set is None:
            return train_scores
        else:
            return train_scores, val_scores

    def score(self, data_set) -> float:
        pass

    def _fit(self, x, y):
        pass
