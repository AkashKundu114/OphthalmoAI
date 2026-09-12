# Contributing to OphthalmoAI

First off, thank you for considering contributing to OphthalmoAI! It's people like you that make the open-source community such an amazing place to learn, inspire, and create.

## Where do I go from here?

If you've noticed a bug or have a feature request, make sure to check our [Issues](../../issues) to see if someone else has already created a ticket. If not, go ahead and make one!

## Setting up your development environment

1. **Fork** the repo on GitHub
2. **Clone** the project to your own machine
3. **Install** the requirements:
   - For the backend: `cd backend && pip install -r requirements.txt`
   - For the frontend: `cd frontend && npm install`
4. **Run** the development servers:
   - Backend: `uvicorn backend.main:app --reload --port 8000`
   - Frontend: `cd frontend && npm run dev`
5. **Run tests** before opening a PR:
   - Backend: `pytest tests -q`
   - Frontend: `cd frontend && npm run build`

## Pull Request Process

1. Ensure any install or build dependencies are removed before the end of the layer when doing a build.
2. Update the README.md with details of changes to the interface, this includes new environment variables, exposed ports, useful file locations and container parameters.
3. You may merge the Pull Request in once you have the sign-off of two other developers, or if you do not have permission to do that, you may request the second reviewer to merge it for you.

## Clinical Data & ML Models
When contributing to the machine learning pipeline, ensure that no PHI (Protected Health Information) is included in your commits. All training datasets must be fully anonymized.

Thank you for contributing!
