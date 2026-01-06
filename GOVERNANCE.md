# Project Governance

This document describes the governance structure and decision-making processes for the Unbitrium project.

---

## Table of Contents

1. [Overview](#overview)
2. [Project Lead](#project-lead)
3. [Roles and Responsibilities](#roles-and-responsibilities)
4. [Decision Making](#decision-making)
5. [Contribution Process](#contribution-process)
6. [Conflict Resolution](#conflict-resolution)
7. [Changes to Governance](#changes-to-governance)

---

## Overview

Unbitrium is an open-source federated learning simulator maintained by the Technical University of Denmark (DTU). The project follows a Benevolent Dictator For Life (BDFL) governance model, with the project lead having final authority on all decisions while actively seeking community input.

---

## Project Lead

### Current Lead

| Attribute | Value |
|-----------|-------|
| Name | Olaf Yunus Laitinen Imanov |
| Email | <oyli@dtu.dk> |
| Role | Benevolent Dictator For Life (BDFL) |
| Institution | Technical University of Denmark (DTU) |
| Department | DTU Compute |

### Responsibilities

The project lead is responsible for:

1. **Technical Direction**
   - Setting the project roadmap and priorities
   - Defining architectural decisions
   - Approving major feature additions

2. **Quality Assurance**
   - Reviewing and approving all pull requests
   - Maintaining code quality standards
   - Ensuring test coverage requirements

3. **Community Management**
   - Responding to issues and discussions
   - Mentoring contributors
   - Representing the project externally

4. **Release Management**
   - Planning and executing releases
   - Maintaining the changelog
   - Ensuring backward compatibility

5. **Governance**
   - Enforcing the Code of Conduct
   - Resolving conflicts
   - Making final decisions on disputes

---

## Roles and Responsibilities

### Contributors

Anyone who contributes code, documentation, or other improvements to the project.

| Privilege | Description |
|-----------|-------------|
| Fork and PR | Submit pull requests for review |
| Issue creation | Report bugs and request features |
| Discussion | Participate in project discussions |

### Core Contributors

Trusted contributors with a track record of quality contributions.

| Privilege | Description |
|-----------|-------------|
| Issue triage | Label and organize issues |
| Code review | Review PRs (non-binding) |
| Mentorship | Help new contributors |

*Note: There are currently no core contributors. This role will be established as the community grows.*

### Maintainers

Individuals with write access to the repository.

| Privilege | Description |
|-----------|-------------|
| Merge PRs | Merge approved pull requests |
| Branch management | Create and delete branches |
| Release preparation | Prepare release commits |

*Note: The project lead is currently the only maintainer.*

---

## Decision Making

### Types of Decisions

| Type | Process | Examples |
|------|---------|----------|
| Minor | Project lead decides | Bug fixes, minor docs updates |
| Standard | RFC with community input | New features, API changes |
| Major | Extended RFC and voting | Breaking changes, governance |

### Request for Comments (RFC)

For significant changes, the following RFC process is used:

1. **Proposal**: Open a GitHub Discussion with the proposal
2. **Discussion**: Community feedback period (minimum 7 days)
3. **Revision**: Update proposal based on feedback
4. **Decision**: Project lead makes final decision
5. **Implementation**: Proceed with approved approach

### Voting

For major decisions affecting the project direction:

1. Voting period: 14 days
2. Eligible voters: Core contributors and maintainers
3. Quorum: 50% of eligible voters
4. Outcome: Simple majority, with project lead tie-breaker

---

## Contribution Process

### Code Contributions

1. Fork the repository
2. Create a feature branch
3. Implement changes with tests
4. Submit a pull request
5. Address review feedback
6. Merge upon approval

See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines.

### Documentation Contributions

Documentation follows the same process as code contributions, with the following considerations:

- Technical accuracy is paramount
- API documentation must match implementation
- Tutorial completeness is required (400+ lines)

### Issue Reporting

1. Search existing issues first
2. Use provided templates
3. Include reproducible examples
4. Be responsive to follow-up questions

---

## Conflict Resolution

### Technical Disagreements

1. **Discussion**: Attempt to reach consensus in the PR or issue
2. **Escalation**: If no consensus, escalate to project lead
3. **Decision**: Project lead makes final decision with reasoning

### Code of Conduct Violations

1. **Report**: Email project lead directly
2. **Investigation**: Review of reported behavior
3. **Action**: Appropriate response per severity
4. **Appeal**: Appeals may be made to the project lead

See [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) for details.

---

## Changes to Governance

This governance document may be updated through the following process:

1. **Proposal**: Submit changes as a pull request
2. **Discussion**: Minimum 14-day community feedback period
3. **Voting**: Core contributors and maintainers vote
4. **Approval**: Requires 2/3 majority
5. **Merge**: Project lead merges approved changes

---

## Transparency

All governance decisions are documented in:

- GitHub Issues (for discussions)
- Pull requests (for changes)
- CHANGELOG.md (for release decisions)
- This document (for process updates)

---

## Contact

For governance-related questions, contact the project lead at <oyli@dtu.dk>.

---

*Last updated: January 2026*
