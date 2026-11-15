# Contributing

## Before you start

```bash
git switch main
git pull origin main
git switch -c feat/short-description
```

## While working

- Keep each branch focused on one task.
- Coordinate changes to `contracts.py` before editing it.
- Do not commit secrets, restricted data, or personal credentials.
- Use English for code comments, documentation, commit messages, issues, and pull requests.

## Before opening a pull request

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
git status
git add <files-you-changed>
git commit -m "Add <short description>"
git push -u origin feat/short-description
```

Open a Pull Request from your branch into `main`. Explain what changed, how it was tested, and whether another module depends on the change.
