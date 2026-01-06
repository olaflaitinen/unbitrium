# Security Policy

This document outlines the security policies and procedures for the Unbitrium project.

---

## Table of Contents

1. [Supported Versions](#supported-versions)
2. [Reporting a Vulnerability](#reporting-a-vulnerability)
3. [Security Update Process](#security-update-process)
4. [Security Best Practices](#security-best-practices)
5. [Dependency Management](#dependency-management)
6. [Known Limitations](#known-limitations)

---

## Supported Versions

The following versions of Unbitrium are currently supported with security updates:

| Version | Supported | End of Life |
|---------|-----------|-------------|
| 1.0.x   | Yes       | TBD         |
| < 1.0   | No        | N/A         |

Security patches are released as point releases (e.g., 1.0.1) and are backward compatible within the same major version.

---

## Reporting a Vulnerability

### Do NOT Open Public Issues

Security vulnerabilities must NOT be reported through public GitHub issues. Public disclosure of vulnerabilities puts users at risk.

### Reporting Process

1. **Email the project lead directly**:

   | Contact | Email |
   |---------|-------|
   | Olaf Yunus Laitinen Imanov | <oyli@dtu.dk> |

2. **Subject line**: `[SECURITY] Brief description of vulnerability`

3. **Include the following information**:

   | Information | Description |
   |-------------|-------------|
   | Type | Category of vulnerability (e.g., injection, privacy leak) |
   | Component | Affected module or function |
   | Location | File path and line numbers if known |
   | Version | Affected version(s) |
   | Impact | Potential consequences of exploitation |
   | Reproduction | Step-by-step instructions to reproduce |
   | PoC | Proof-of-concept code if available |

4. **Encryption** (optional but recommended): Use PGP encryption for sensitive reports.

### Response Timeline

| Phase | Timeframe |
|-------|-----------|
| Acknowledgment | Within 48 hours |
| Initial assessment | Within 7 days |
| Fix development | Depends on severity |
| Disclosure | After fix is released |

### Severity Classification

| Severity | Description | Response Time |
|----------|-------------|---------------|
| Critical | Remote code execution, data exfiltration | 48 hours |
| High | Significant privacy breach, DoS | 7 days |
| Medium | Limited information disclosure | 14 days |
| Low | Minor issues, hardening | 30 days |

---

## Security Update Process

### Patch Release

1. Security fix is developed and tested
2. New version is released with fix
3. Security advisory is published
4. Affected users are notified via GitHub Security Advisories

### Disclosure Policy

We follow coordinated disclosure:

1. Reporter is credited (unless anonymity is requested)
2. Advisory includes:
   - Description of vulnerability
   - Affected versions
   - Patched versions
   - Mitigation steps
   - Timeline

---

## Security Best Practices

### For Users

| Practice | Description |
|----------|-------------|
| Update regularly | Always use the latest stable version |
| Pin dependencies | Use lockfiles (requirements.txt, poetry.lock) |
| Review configs | Audit configuration files for sensitive data |
| Limit access | Use least-privilege principles |

### For Contributors

| Practice | Description |
|----------|-------------|
| No secrets in code | Never commit credentials or API keys |
| Input validation | Validate all user inputs |
| Safe deserialization | Be cautious with pickle/eval |
| Dependency audit | Check dependencies for known vulnerabilities |

---

## Dependency Management

### Automated Scanning

We use the following tools for dependency security:

| Tool | Purpose |
|------|---------|
| Dependabot | Automated dependency updates |
| pip-audit | Python vulnerability scanning |
| CodeQL | Static analysis |

### Dependency Update Policy

| Type | Frequency |
|------|-----------|
| Security patches | Immediate |
| Minor updates | Monthly |
| Major updates | Quarterly |

---

## Known Limitations

### Privacy Considerations

Unbitrium is a simulation library. While it implements differential privacy mechanisms, users should be aware:

1. **Simulation vs. Production**: Privacy mechanisms are for research simulation, not production deployment with real sensitive data.

2. **Parameter Selection**: Privacy guarantees depend on correct parameter selection (epsilon, delta).

3. **Side Channels**: The library does not protect against timing or other side-channel attacks.

### Scope of Security

This security policy covers:
- Core library code (`src/unbitrium/`)
- Official examples and benchmarks
- Configuration files

This security policy does NOT cover:
- Third-party code using Unbitrium
- Deployment infrastructure
- User-generated content

---

## Acknowledgments

We thank the following individuals for responsibly disclosing security issues:

*No security vulnerabilities have been reported yet.*

---

## Contact

For security-related communications:

- **Email**: <oyli@dtu.dk>
- **Subject**: `[SECURITY] Your Subject`

---

*Last updated: January 2026*
