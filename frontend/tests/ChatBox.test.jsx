import { describe, it, expect } from 'vitest';
import { render, screen } from '@testing-library/react';
import ChatBot from '../src/ChatBox';

describe('ChatBox Component', () => {
  it('renders without crashing', () => {
    render(<ChatBot chatHistory={[]} />);
    expect(screen.getByText(/Ask follow-up questions/i)).toBeInTheDocument();
  });
});
