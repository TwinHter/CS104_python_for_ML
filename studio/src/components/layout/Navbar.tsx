
"use client";
import Link from 'next/link';
import { usePathname } from 'next/navigation';
import Logo from './Logo';
import { cn } from '@/lib/utils';
import { Button } from '@/components/ui/button'; 
import { NAV_ITEMS } from '@/lib/constants';

const Navbar = () => {
  const pathname = usePathname();

  return (
    <nav className="bg-card text-card-foreground shadow-md sticky top-0 z-50">
      <div className="container mx-auto px-4 py-3 flex justify-between items-center">
        <Logo />
        <ul className="flex space-x-1 sm:space-x-2 items-center">
          {NAV_ITEMS.map((item) => (
            <li key={item.href}>
              <Button
                variant="ghost"
                asChild
                className={cn(
                  "text-sm sm:text-base font-medium transition-colors duration-150 ease-in-out px-3 py-2 rounded-md",
                  pathname === item.href 
                    ? "bg-accent text-accent-foreground hover:bg-accent/90" // Active state
                    : "text-primary hover:text-primary hover:bg-accent/20" // Inactive state
                )}
              >
                <Link href={item.href}>{item.label}</Link>
              </Button>
            </li>
          ))}
        </ul>
      </div>
    </nav>
  );
};

export default Navbar;
