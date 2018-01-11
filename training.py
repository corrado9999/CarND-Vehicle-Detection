import os
import ast
import glob
import numpy as np
import cv2
import click

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.externals.joblib import dump

import lesson_functions
import features
import training

def get_dataset(*glob_expr, flip=False, limit_per_expr=None, split=False, **split_kwargs):
    data = [
        (np.stack(cv2.imread(f)), klass)
        for klass, g in enumerate(glob_expr)
        for f in glob.glob(g)[:limit_per_expr]
    ]
    if flip:
        data.extend([(img[..., ::-1, :], klass) for img,klass in data])
    x, y = tuple(zip(*data))
    if not split:
        return np.array(x), np.array(y)
    else:
        return train_test_split(x, y, **split_kwargs)

def get_base_model(memory='./sklearn-cache'):
    classifier = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC()),
    ], memory=memory)

    pipeline = Pipeline([
        ("features", features.FeatureExtraction()),
        ('classifier', classifier),
    ], memory=memory)
    return pipeline

def select_model(base_model, param_grid, x, y, n_jobs=1, **grid_search_kwargs):
    grid_search = GridSearchCV(base_model, param_grid,
                               scoring='accuracy', n_jobs=n_jobs,
                               **grid_search_kwargs)
    grid_search.fit(x, y)
    return grid_search.cv_results_, grid_search.best_estimator_, \
           grid_search.best_score_, grid_search.best_params_

@click.command()
@click.argument('output-path')
@click.argument('input-globs', nargs=-1)
@click.option('--param-grid', type=ast.literal_eval, default='{}')
@click.option('--n-jobs', '-j', type=int, default=1)
@click.option('--search-options', type=ast.literal_eval, default='{}')
@click.option('--verbose', '-v', count=True)
@click.option('--cache-dir', '-c', default='sklearn-cache')
def main(output_path, input_globs, param_grid={}, n_jobs=1, search_options={},
         verbose=0, cache_dir='sklearn-cache'):
    if not os.path.exists(cache_dir):
        os.mkdir(cache_dir)

    print("Reading dataset")
    x, y = get_dataset(*input_globs)
    model = get_base_model(memory=cache_dir)
    print("Starting training")
    result, best_model, best_score, best_params = select_model(model, param_grid,
                                                               x, y,
                                                               n_jobs=n_jobs,
                                                               verbose=verbose,
                                                               **search_options)
    print("Successfully trained model with accuracy: {:.1%}".format(best_score))
    print("Best hyper-parameters combination:\n%r" % best_params)

    param_id = '_'.join(
        k.replace('__', '-') + '=' + str(v)
        for k,v in sorted(best_params.items()))
    outfile = output_path.format(best_params=param_id, **best_params)

    print("Writing output file: %r" % outfile)
    dump(dict(result=result,
              best_model=best_model,
              best_score=best_score,
              best_params=best_params),
         outfile)

if __name__ == '__main__':
    main()
