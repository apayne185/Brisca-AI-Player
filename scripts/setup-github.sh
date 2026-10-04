#!/usr/bin/env bash
# Apply repository settings and branch protection as code.
# Idempotent: safe to re-run after editing .github/rulesets/main.json.
#
# Requires the GitHub CLI authenticated as a repo admin: `gh auth login`.
set -euo pipefail

repo="${1:-$(gh repo view --json nameWithOwner -q .nameWithOwner)}"
ruleset_file="$(dirname "$0")/../.github/rulesets/main.json"
ruleset_name="$(jq -r .name "$ruleset_file")"

echo "Configuring ${repo}"

# Squash-only merges with the PR title as the commit message, tidy branches,
# and allow auto-merge so Dependabot PRs land once CI is green.
gh api -X PATCH "repos/${repo}" --silent \
  -f description="Imperfect-information game AI platform for the Spanish card game Brisca." \
  -F has_wiki=false \
  -F allow_squash_merge=true \
  -F allow_merge_commit=false \
  -F allow_rebase_merge=false \
  -F allow_auto_merge=true \
  -F delete_branch_on_merge=true \
  -F allow_update_branch=true \
  -f squash_merge_commit_title=PR_TITLE \
  -f squash_merge_commit_message=PR_BODY

gh api -X PUT "repos/${repo}/topics" --silent \
  -f 'names[]=machine-learning' -f 'names[]=reinforcement-learning' \
  -f 'names[]=game-ai' -f 'names[]=mcts' -f 'names[]=imperfect-information' \
  -f 'names[]=brisca' -f 'names[]=python'

# Dependabot alerts and security updates.
gh api -X PUT "repos/${repo}/vulnerability-alerts" --silent
gh api -X PUT "repos/${repo}/automated-security-fixes" --silent

# Create or update the branch ruleset.
existing_id="$(gh api "repos/${repo}/rulesets" -q ".[] | select(.name == \"${ruleset_name}\") | .id")"
if [[ -n "${existing_id}" ]]; then
  gh api -X PUT "repos/${repo}/rulesets/${existing_id}" --input "$ruleset_file" --silent
  echo "Updated ruleset '${ruleset_name}' (${existing_id})"
else
  gh api -X POST "repos/${repo}/rulesets" --input "$ruleset_file" --silent
  echo "Created ruleset '${ruleset_name}'"
fi

echo "Done."
