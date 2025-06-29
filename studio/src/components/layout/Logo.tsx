
import Link from 'next/link';
import { Landmark } from 'lucide-react'; // Import the Landmark icon
import { APP_NAME } from '@/lib/constants';

const Logo = () => {
  return (
    <Link href="/" className="flex items-center text-2xl font-bold font-headline hover:opacity-90 transition-opacity text-primary">
      <Landmark className="w-7 h-7 mr-2" /> {/* Add the icon here */}
      <span>{APP_NAME}</span>
    </Link>
  );
};

export default Logo;
