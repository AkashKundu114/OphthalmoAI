import { describe, it, expect } from 'vitest'
import { render, screen, fireEvent } from '@testing-library/react'
import ClinicalResearchPage, { PAPERS_DATABASE } from '../src/ClinicalResearchPage'
import App from '../src/App'

describe('Clinical Research & Preprints Page', () => {
  it('renders research database with arXiv preprints and journal articles', () => {
    render(<ClinicalResearchPage />)

    // Header
    expect(screen.getByRole('heading', { name: /Clinical Research & Preprints/i })).toBeInTheDocument()

    // arXiv preprints
    expect(screen.getByText(/RETFound-Green/i)).toBeInTheDocument()
    expect(screen.getByText(/Dual-IFM/i)).toBeInTheDocument()
    expect(screen.getByText(/Heterogeneous Meta-Classifier Ensemble/i)).toBeInTheDocument()

    // Peer-reviewed papers
    expect(screen.getAllByText(/Nature Medicine/i).length).toBeGreaterThan(0)
    expect(screen.getAllByText(/The Lancet Global Health/i).length).toBeGreaterThan(0)
  })

  it('filters publications by publication type (arXiv vs Journal)', () => {
    render(<ClinicalResearchPage />)

    // Click arXiv preprints filter tab
    const arxivTab = screen.getByRole('button', { name: /arXiv Preprints/i })
    fireEvent.click(arxivTab)

    // Should show arXiv papers
    expect(screen.getByText(/RETFound-Green/i)).toBeInTheDocument()
    // Should NOT show Lancet Myopia paper under arXiv tab
    expect(screen.queryByText(/The Lancet Global Health/i)).not.toBeInTheDocument()

    // Click Peer-Reviewed Journals tab
    const journalTab = screen.getByRole('button', { name: /Peer-Reviewed Journals/i })
    fireEvent.click(journalTab)

    // Should show journal paper
    expect(screen.getByText(/The Lancet Global Health/i)).toBeInTheDocument()
    // Should NOT show RETFound-Green under Journal tab
    expect(screen.queryByText(/RETFound-Green/i)).not.toBeInTheDocument()
  })

  it('filters publications using the real-time search input', () => {
    render(<ClinicalResearchPage />)

    const searchInput = screen.getByPlaceholderText(/Search arXiv ID, author, model, topic/i)
    fireEvent.change(searchInput, { target: { value: 'Conformal' } })

    expect(screen.getByText(/Uncertainty-Aware Deep Learning with Conformal Prediction/i)).toBeInTheDocument()
    expect(screen.queryByText(/The Lancet Global Health/i)).not.toBeInTheDocument()
  })

  it('navigates to Clinical Research page from App header and footer', () => {
    render(<App />)

    // Click Clinical Research tab in navigation
    const researchNavButtons = screen.getAllByRole('button', { name: /Clinical Research/i })
    expect(researchNavButtons.length).toBeGreaterThan(0)
    fireEvent.click(researchNavButtons[0])

    expect(screen.getByRole('heading', { name: /Clinical Research & Preprints/i })).toBeInTheDocument()
  })
})
