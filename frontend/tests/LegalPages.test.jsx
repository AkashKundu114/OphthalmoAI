import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen, fireEvent } from '@testing-library/react'
import TermsPage from '../src/TermsPage'
import PrivacyPolicyPage from '../src/PrivacyPolicyPage'
import App from '../src/App'

beforeEach(() => {
  window.scrollTo = vi.fn()
})

describe('Terms & Conditions Page', () => {
  it('renders the terms title, medical disclaimer and core sections', () => {
    const onNavigate = vi.fn()
    render(<TermsPage onNavigate={onNavigate} />)

    // Heading
    expect(screen.getByRole('heading', { name: /Terms & Conditions/i })).toBeInTheDocument()

    // Medical disclaimer
    expect(screen.getAllByText(/Medical AI & Clinical Use Disclaimer/i).length).toBeGreaterThan(0)
    expect(screen.getByText(/Auxiliary Decision-Support Only — Not Autonomous Medical Care/i)).toBeInTheDocument()

    // Acceptable use and patient consent
    expect(screen.getAllByText(/Acceptable Use & User Qualifications/i).length).toBeGreaterThan(0)
    expect(screen.getAllByText(/Patient Consent, HIPAA & De-Identification/i).length).toBeGreaterThan(0)

    // Navigation trigger
    const backBtn = screen.getByRole('button', { name: /Back to Diagnostic Screening/i })
    fireEvent.click(backBtn)
    expect(onNavigate).toHaveBeenCalledWith('diagnostic')
  })

  it('filters sections when clicking filter pills', () => {
    const onNavigate = vi.fn()
    render(<TermsPage onNavigate={onNavigate} />)

    // Click on Intellectual Property filter
    const ipBtn = screen.getByRole('button', { name: /Intellectual Property & Model Weights/i })
    fireEvent.click(ipBtn)

    // Section should still be visible
    expect(screen.getByText(/4\. Intellectual Property & Model Weights/i)).toBeInTheDocument()
  })
})

describe('Privacy Policy Page', () => {
  it('renders privacy philosophy, HIPAA/GDPR standards and trust badges', () => {
    const onNavigate = vi.fn()
    render(<PrivacyPolicyPage onNavigate={onNavigate} />)

    // Title
    expect(screen.getByRole('heading', { name: /Privacy Policy/i })).toBeInTheDocument()

    // Trust badges
    expect(screen.getByText(/Zero Ad Monetization/i)).toBeInTheDocument()
    expect(screen.getByText(/HIPAA De-Identified/i)).toBeInTheDocument()
    expect(screen.getAllByText(/TLS 1\.3 Transport/i).length).toBeGreaterThan(0)

    // Sections
    expect(screen.getAllByText(/Privacy Philosophy & Clinical Data Ethics/i).length).toBeGreaterThan(0)
    expect(screen.getAllByText(/HIPAA, GDPR & Patient De-Identification Standards/i).length).toBeGreaterThan(0)
    expect(screen.getAllByText(/Technical Security & Encryption Architecture/i).length).toBeGreaterThan(0)

    // Navigation to terms
    const termsBtn = screen.getByRole('button', { name: /View Terms & Conditions/i })
    fireEvent.click(termsBtn)
    expect(onNavigate).toHaveBeenCalledWith('terms')
  })
})

describe('App Navigation to Legal Pages', () => {
  it('navigates to Terms & Conditions and Privacy Policy from footer', () => {
    render(<App />)

    // Find Terms & Conditions in footer
    const termsButtons = screen.getAllByRole('button', { name: /Terms & Conditions/i })
    expect(termsButtons.length).toBeGreaterThan(0)
    fireEvent.click(termsButtons[0])

    // Should now show Terms & Conditions page
    expect(screen.getByRole('heading', { name: /Terms & Conditions/i })).toBeInTheDocument()

    // Find Privacy Policy in footer
    const privacyButtons = screen.getAllByRole('button', { name: /Privacy Policy/i })
    expect(privacyButtons.length).toBeGreaterThan(0)
    fireEvent.click(privacyButtons[0])

    // Should now show Privacy Policy page
    expect(screen.getByRole('heading', { name: /Privacy Policy/i })).toBeInTheDocument()
  })
})
