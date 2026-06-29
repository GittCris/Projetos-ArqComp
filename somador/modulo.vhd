library IEEE;
use IEEE.STD_LOGIC_1164.ALL;
use IEEE.NUMERIC_STD.ALL;

entity SomadorSubtratorVetorial is
    Port (
        A_i       : in  STD_LOGIC_VECTOR(31 downto 0);
        B_i       : in  STD_LOGIC_VECTOR(31 downto 0);
        mode_i    : in  STD_LOGIC;
        vecSize_i : in  STD_LOGIC_VECTOR(1 downto 0);
        S_o       : out STD_LOGIC_VECTOR(31 downto 0)
    );
end SomadorSubtratorVetorial;

architecture Behavioral of SomadorSubtratorVetorial is
    signal B_mux : STD_LOGIC_VECTOR(31 downto 0);
    signal carry : STD_LOGIC_VECTOR(32 downto 0);
begin

    gen_b_mux: for i in 0 to 31 generate
        B_mux(i) <= B_i(i) xor mode_i;
    end generate;

    process(A_i, B_mux, mode_i, vecSize_i, carry)
    begin
        carry(0) <= mode_i;

        for i in 0 to 31 loop
            carry(i+1) <= (A_i(i) and B_mux(i)) or (carry(i) and (A_i(i) xor B_mux(i)));

            if vecSize_i = "00" and ((i + 1) mod 4 = 0) then
                carry(i+1) <= mode_i;
            elsif vecSize_i = "01" and ((i + 1) mod 8 = 0) then
                carry(i+1) <= mode_i;
            elsif vecSize_i = "10" and ((i + 1) mod 16 = 0) then
                carry(i+1) <= mode_i;
            end if;
        end loop;
    end process;

    gen_saida: for i in 0 to 31 generate
        S_o(i) <= A_i(i) xor B_mux(i) xor carry(i);
    end generate;

end Behavioral;