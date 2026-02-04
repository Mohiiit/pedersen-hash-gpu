# Vast.ai CLI (Notes for GPU Testing)

This is a quick, repo-local summary of the Vast.ai CLI commands we’ll use to rent and connect to a GPU instance for testing.

## Install

- PyPI install:
  - `pip install vastai`
- Latest script from GitHub:
  - `wget https://raw.githubusercontent.com/vast-ai/vast-python/master/vast.py -O vast; chmod +x vast;`

## Auth

- Set API key (from the Vast.ai console CLI page):
  - `vastai set api-key <YOUR_API_KEY>`
  - This stores the key in a local config file so you don’t need to pass it on every command.

## Discover Offers

- List help for search filters:
  - `vastai search offers --help`
- Example search for compute capability >= 8.0:
  - `vastai search offers 'compute_cap >= 800'`

## Create / Launch Instance

- Create instance from an offer ID:
  - `vastai create instance <OFFER_ID> --image <IMAGE> --disk <GB> --ssh --direct`
  - Note: `--image` is required for create instance.

## Connect (SSH)

- The CLI includes:
  - `attach ssh` (attach your SSH key to an instance)
  - `ssh-url` (prints an SSH connection helper for an instance)
- Use `--help` for exact args:
  - `vastai attach ssh --help`
  - `vastai ssh-url --help`

## Inspect / Manage

- Show instances:
  - `vastai show instances`
- Show a specific instance:
  - `vastai show instance <ID>`
- Start/stop:
  - `vastai start instance <ID>`
  - `vastai stop instance <ID>`

## Data Copy

- Copy data between local and instance paths:
  - `vastai copy <SRC> <DST>`
  - `SRC`/`DST` support multiple formats (see `vastai copy --help`).

## Notes for Our Use

- We can use the CLI to provision a GPU instance, attach SSH, and connect to run our CUDA tests.
- Once connected, run the GPU test from this repo:
  - `CUDA_ARCH=sm_86 cargo test --features cuda -- --ignored`
  - Adjust `CUDA_ARCH` for the instance GPU model.
