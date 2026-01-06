# Security Policy

This document outlines the security policies and procedures for the Unbitrium project, a production-grade Federated Learning Simulator developed at the Technical University of Denmark.

## Table of Contents

1. [Supported Versions](#supported-versions)
2. [Reporting a Vulnerability](#reporting-a-vulnerability)
3. [Vulnerability Disclosure Process](#vulnerability-disclosure-process)
4. [Security Response Timeline](#security-response-timeline)
5. [Security Best Practices](#security-best-practices)
6. [Dependency Management](#dependency-management)
7. [Scope](#scope)
8. [Acknowledgments](#acknowledgments)

---

## Supported Versions

The following versions of Unbitrium receive security updates:

| Version | Support Status | End of Support |
|---------|----------------|----------------|
| 1.0.x   | Active         | To be determined |
| < 1.0   | Unsupported    | Not applicable |

Security patches are released as point releases (e.g., 1.0.1, 1.0.2) and maintain backward compatibility within the same major version. Users are strongly encouraged to update to the latest patch release.

---

## Reporting a Vulnerability

### Contact Information

To report a security vulnerability, please contact the project maintainer directly:

- **Email**: oyli@dtu.dk
- **Subject Line**: [SECURITY] Brief description of the vulnerability

### Reporting Guidelines

When submitting a vulnerability report, please include the following information:

1. **Vulnerability Type**: Classification of the vulnerability (e.g., remote code execution, information disclosure, denial of service, injection attack).

2. **Affected Component**: The specific module, function, or file containing the vulnerability.

3. **Location**: File path, line numbers, and relevant code snippets if available.

4. **Affected Versions**: List of versions known to be affected.

5. **Reproduction Steps**: Detailed, step-by-step instructions to reproduce the vulnerability.

6. **Proof of Concept**: Code, scripts, or other materials demonstrating the vulnerability.

7. **Impact Assessment**: Description of the potential security impact and attack scenarios.

8. **Suggested Remediation**: If available, proposed fixes or mitigation strategies.

### Confidentiality

Please do NOT disclose the vulnerability publicly through GitHub Issues, pull requests, discussion forums, social media, or other public channels until a fix has been released and an appropriate disclosure period has elapsed.

---

## Vulnerability Disclosure Process

The Unbitrium project follows coordinated vulnerability disclosure practices in accordance with industry standards.

### Phase 1: Receipt and Acknowledgment

Upon receiving a vulnerability report, the security team will:

- Acknowledge receipt within 48 hours
- Assign a unique tracking identifier
- Designate a primary contact for the reporter

### Phase 2: Assessment and Triage

The security team will:

- Verify the reported vulnerability
- Assess severity using the Common Vulnerability Scoring System (CVSS)
- Determine affected versions and potential impact
- Communicate initial findings to the reporter within 7 days

### Phase 3: Remediation

Based on the severity assessment:

- **Critical**: Fix developed and released within 30 days
- **High**: Fix developed and released within 60 days
- **Medium**: Fix developed and released within 90 days
- **Low**: Fix scheduled for the next regular release cycle

### Phase 4: Disclosure

Following the release of a security fix:

- A GitHub Security Advisory will be published
- The reporter will be credited unless anonymity is requested
- CVE identifiers will be requested for significant vulnerabilities
- Release notes will document the security fix

---

## Security Response Timeline

| Phase | Timeframe | Description |
|-------|-----------|-------------|
| Acknowledgment | 48 hours | Confirmation of report receipt |
| Initial Assessment | 7 days | Severity evaluation and verification |
| Critical Fix | 30 days | Remediation for critical vulnerabilities |
| High Severity Fix | 60 days | Remediation for high severity issues |
| Medium/Low Fix | 90 days | Remediation for medium and low severity issues |
| Public Disclosure | Post-fix | After patch is available to users |

---

## Security Best Practices

### For Users

- **Version Management**: Always use the latest stable release to benefit from security patches.
- **Dependency Pinning**: Use lockfiles (e.g., `requirements.txt`, `poetry.lock`) to ensure reproducible and auditable installations.
- **Configuration Security**: Avoid committing sensitive configuration data to version control.
- **Least Privilege**: Run the software with minimal necessary permissions.

### For Contributors

- **Secure Coding**: Follow secure coding guidelines and avoid known vulnerability patterns.
- **Input Validation**: Validate and sanitize all external inputs.
- **Dependency Awareness**: Be mindful of the security posture of added dependencies.
- **Secret Management**: Never commit credentials, API keys, or other secrets to the repository.
- **Safe Deserialization**: Exercise caution with pickle, eval, and other deserialization mechanisms.

---

## Dependency Management

### Automated Security Scanning

The project employs the following security tools:

| Tool | Purpose |
|------|---------|
| Dependabot | Automated dependency updates and vulnerability alerts |
| CodeQL | Static application security testing (SAST) |
| OSV-Scanner | Open source vulnerability detection |
| pip-audit | Python dependency vulnerability scanning |

### Dependency Update Policy

| Update Type | Frequency |
|-------------|-----------|
| Security Patches | Immediate upon availability |
| Minor Updates | Monthly review cycle |
| Major Updates | Quarterly review cycle |

---

## Scope

### In Scope

The following components are covered by this security policy:

- Core library source code (`src/unbitrium/`)
- Official examples and benchmarks
- GitHub Actions workflows and CI/CD configurations
- Docker container images
- Documentation that may affect security

### Out of Scope

The following are not covered by this security policy:

- Third-party applications built using Unbitrium
- User-deployed infrastructure and configurations
- Issues in dependencies (report upstream)
- Social engineering or phishing attacks
- Attacks requiring physical access to systems

---

## Acknowledgments

We appreciate the security research community's efforts in identifying and responsibly disclosing vulnerabilities. Contributors who report security issues will be acknowledged in release notes and security advisories unless anonymity is requested.

---

## Additional Resources

- GitHub Security Advisories: https://github.com/olaflaitinen/unbitrium/security/advisories
- Issue Tracker: https://github.com/olaflaitinen/unbitrium/issues
- Documentation: https://olaflaitinen.github.io/unbitrium

---

*Last Updated: January 2026*

*This security policy is subject to revision. Please check for updates periodically.*
