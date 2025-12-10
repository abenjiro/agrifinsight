import { ReactNode, memo } from 'react'
import Header from './Header'
import Footer from './Footer'

interface LandingLayoutProps {
  children: ReactNode
}

export const LandingLayout = memo(function LandingLayout({ children }: LandingLayoutProps) {
  return (
    <div className="min-h-screen flex flex-col">
      <Header />
      <main className="flex-1">
        {children}
      </main>
      <Footer />
    </div>
  )
})

