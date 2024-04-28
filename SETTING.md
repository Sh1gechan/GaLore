# experiment_galore

# 環境構築
## pyenvのインストール
```
curl https://pyenv.run | bash
```

## 環境変数の設定
```
export PYENV_ROOT="$HOME/.pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init --path)"

. ~/.bashrc
```

## pythonのインストール

```
pyenv install 3.8.5
pyenv local 3.8.5
python -m venv .env
source .env/bin/activate
```

```
pip install -U transformers
pip install -e .

pip install -r exp_requirements.txt
```