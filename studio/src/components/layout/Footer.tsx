import { Github } from 'lucide-react';
import Link from 'next/link';

const Footer = () => {
  return (
    <footer className="bg-muted text-muted-foreground py-8 mt-auto">
      <div className="container mx-auto px-4 text-center">
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-4">
          <div>
            <h3 className="font-headline text-lg text-foreground mb-2">Contact Us</h3>
            <p>contact@londondwellings.ai</p>
            <p>+44 123 456 7890</p>
          </div>
          <div>
            <h3 className="font-headline text-lg text-foreground mb-2">Development Team</h3>
            <p>AI Enthusiasts & Developers</p>
            <p>Firebase Studio Team</p>
          </div>
          <div>
            <h3 className="font-headline text-lg text-foreground mb-2">Resources</h3>
            <Link href="https://github.com" target="_blank" rel="noopener noreferrer" className="flex items-center justify-center hover:text-primary transition-colors">
              <Github className="w-5 h-5 mr-2" />
              <span>GitHub Repository</span>
            </Link>
          </div>
        </div>
        <p className="text-sm">
          &copy; {new Date().getFullYear()} London Dwellings AI. All rights reserved.
        </p>
      </div>
    </footer>
  );
};

export default Footer;
